use anyhow::{Context, Result, anyhow};
use arrow::array::{Array, ArrayRef, Float32Array, ListArray, StringArray};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use dotenv::dotenv;
use log::{error, info};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, UpsertPointsBuilder, VectorParamsBuilder,
};
use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Parser, Debug)]
#[clap(
    name = "qdrant-uploader",
    about = "Upload embeddings from parquet file to Qdrant"
)]
struct Args {
    #[clap(long, required = true, help = "Path to parquet file with embeddings")]
    dataset: String,
    #[clap(long, required = false, help = "Ask for each dataset upload")]
    ask: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init_from_env(
        env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "info"),
    );
    dotenv().ok();

    let args = Args::parse();
    info!("Starting the Qdrant uploader");

    let file = File::open(&args.dataset)
        .with_context(|| format!("Failed to open file {}", args.dataset))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let record_batches: Vec<RecordBatch> = reader.collect::<Result<_, _>>()?;
    if record_batches.is_empty() {
        return Err(anyhow!("Parquet file contains no record batches"));
    }

    let schema = record_batches[0].schema();
    let column_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

    let embedding_columns: Vec<String> = column_names
        .iter()
        .filter(|name| name.starts_with("text_"))
        .cloned()
        .collect();

    info!(
        "Found {} embedding models to process",
        embedding_columns.len()
    );

    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
    info!("Connecting to qdrant: {}", qdrant_url);

    let client = Qdrant::from_url(&qdrant_url)
        .api_key(env::var("QDRANT_API_KEY").ok())
        .build()?;

    for embed_col in &embedding_columns {
        if args.ask {
            print!("Upload '{}' model? (y/n): ", embed_col);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if !input.trim().eq_ignore_ascii_case("y") {
                info!("Skipped {} model", embed_col);
                continue;
            }
        }

        let model_name = embed_col
            .replace("text_", "")
            .replace("/", "_")
            .replace("-", "_");
        let collection_prefix =
            env::var("COLLECTION_PREFIX").unwrap_or_else(|_| "moskva".to_string());
        let collection_name = format!("{}_{}", collection_prefix, model_name);

        info!("Processing data for collection '{}'", collection_name);

        let mut vector_size: u64 = 0;
        let mut found_valid = false;

        'outer: for batch in &record_batches {
            let col_idx = batch.schema().index_of(embed_col)?;
            let array = batch.column(col_idx);
            if let Some(list_array) = array.as_any().downcast_ref::<ListArray>() {
                for i in 0..list_array.len() {
                    if list_array.is_valid(i) {
                        let values = list_array.value(i);
                        if let Some(float_array) = values.as_any().downcast_ref::<Float32Array>() {
                            if float_array.len() > 0 {
                                vector_size = float_array.len() as u64;
                                found_valid = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        if !found_valid {
            error!("No valid embeddings found in column '{}'", embed_col);
            continue;
        }

        info!("Vector size determined: {}", vector_size);

        let collections = client.list_collections().await?;
        let collection_exists = collections
            .collections
            .iter()
            .any(|c| c.name == collection_name);

        if !collection_exists {
            info!("Creating collection '{}'", collection_name);
            client
                .create_collection(
                    CreateCollectionBuilder::new(&collection_name)
                        .vectors_config(VectorParamsBuilder::new(vector_size, Distance::Cosine)),
                )
                .await?;
        }

        let batch_size = env::var("BATCH_SIZE")
            .unwrap_or_else(|_| "100".to_string())
            .parse::<usize>()
            .unwrap_or(100);

        let mut processed = 0;
        let mut skipped = 0;
        let mut points = Vec::with_capacity(batch_size);
        let mut counter: u64 = 0;
        let start_time = Instant::now();

        for batch in &record_batches {
            let embed_idx = batch.schema().index_of(embed_col)?;
            let address_idx = batch.schema().index_of("address").unwrap_or(usize::MAX);
            let name_idx = batch.schema().index_of("name_ru").unwrap_or(usize::MAX);
            let rating_idx = batch.schema().index_of("rating").unwrap_or(usize::MAX);
            let rubrics_idx = batch.schema().index_of("rubrics").unwrap_or(usize::MAX);
            let text_idx = batch.schema().index_of("text").unwrap_or(usize::MAX);

            let num_rows = batch.num_rows();

            for row_idx in 0..num_rows {
                let mut valid_point = true;
                let mut vector = Vec::new();

                if let Some(list_array) =
                    batch.column(embed_idx).as_any().downcast_ref::<ListArray>()
                {
                    if list_array.is_valid(row_idx) {
                        let values = list_array.value(row_idx);
                        if let Some(float_array) = values.as_any().downcast_ref::<Float32Array>() {
                            if float_array.len() > 0 {
                                vector = (0..float_array.len())
                                    .map(|i| float_array.value(i))
                                    .collect();
                            } else {
                                valid_point = false;
                            }
                        } else {
                            valid_point = false;
                        }
                    } else {
                        valid_point = false;
                    }
                } else {
                    valid_point = false;
                }

                if !valid_point {
                    skipped += 1;
                    continue;
                }

                let point = PointStruct::new(
                    counter,
                    vector,
                    [
                        (
                            "address",
                            extract_string(batch.column(address_idx), row_idx).into(),
                        ),
                        (
                            "name",
                            extract_string(batch.column(name_idx), row_idx).into(),
                        ),
                        (
                            "rating",
                            extract_f64(batch.column(rating_idx), row_idx)
                                .unwrap_or(0.0)
                                .into(),
                        ),
                        (
                            "rubrics",
                            extract_string(batch.column(rubrics_idx), row_idx)
                                .split(';')
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect()
                                .into(),
                        ),
                        (
                            "text",
                            extract_string(batch.column(text_idx), row_idx).into(),
                        ),
                    ],
                );

                points.push(point);
                processed += 1;
                counter += 1;

                if points.len() >= batch_size {
                    if let Err(e) = client
                        .upsert_points(UpsertPointsBuilder::new(
                            collection_name.clone(),
                            points.clone(),
                        ))
                        .await
                    {
                        error!("Error uploading batch to '{}': {}", collection_name, e);
                    }

                    points.clear();

                    let elapsed = start_time.elapsed();
                    let rps = processed as f64 / elapsed.as_secs_f64();
                    info!(
                        "Progress: {}/{} points, {:.2} points/sec",
                        processed,
                        processed + skipped,
                        rps
                    );
                }
            }
        }

        if !points.is_empty() {
            if let Err(e) = client
                .upsert_points(UpsertPointsBuilder::new(collection_name.clone(), points))
                .await
            {
                error!(
                    "Error uploading final batch to '{}': {}",
                    collection_name, e
                );
            }
        }

        info!(
            "Uploaded {} points to '{}', skipped {} invalid embeddings",
            processed, collection_name, skipped
        );
    }

    info!("All collections have been processed successfully");
    Ok(())
}

fn extract_string(array: &ArrayRef, row_idx: usize) -> String {
    if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
        if string_array.is_valid(row_idx) {
            return string_array.value(row_idx).to_string();
        }
    }
    String::new()
}

fn extract_f64(array: &ArrayRef, row_idx: usize) -> Option<f64> {
    if let Some(float64_array) = array.as_any().downcast_ref::<arrow::array::Float64Array>() {
        return Some(float64_array.value(row_idx));
    }
    None
}
