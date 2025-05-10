# def mean_pool_sentence_embeddings(texts: list[str], model: SentenceTransformer, prompt: Optional[str]) -> np.ndarray:
#     all_embeddings = []
#     for text in tqdm(texts):
#         sentences = sent_tokenize(text, language="russian")
#         sentence_embeddings = model.encode(sentences, convert_to_numpy=True, batch_size=32, prompt=prompt)
#         mean_embedding = np.mean(sentence_embeddings, axis=0)
#         all_embeddings.append(mean_embedding)
#     return np.array(all_embeddings)


# def generate_st_embeddings(
#     df: pd.DataFrame,
#     doc_col: str,
#     embedding_col: str,
#     model: SentenceTransformer,
#     prompt: Optional[str],
# ) -> pd.DataFrame:
#     if doc_col not in df.columns:
#         logger.error(f"DataFrame must contain '{doc_col}' column")
#         raise ValueError(f"DataFrame must contain '{doc_col}' column")

#     if embedding_col in df.columns:
#         logger.info(f"Embedding column {embedding_col} already exists, skipping")

#     else:
#         logger.info(f"Generating embeddings for {doc_col} and saving to {embedding_col} column")
#         doc_embeddings = model.encode(
#             df[doc_col].to_list(),
#             batch_size=get_dynamic_batch_size(model),
#             prompt=prompt,
#             show_progress_bar=True,
#         )
#         # doc_embeddings = mean_pool_sentence_embeddings(df[doc_col].tolist(), model) # mean_pooling makes results worse
#         df[embedding_col] = list(doc_embeddings)
#     return df


# def generate_st_embeddings_models_df(
#     df: pd.DataFrame, doc_col: str, models_prompts: dict[str, dict[str, Optional[str]]]
# ) -> pd.DataFrame:
#     df[doc_col] = df[doc_col].apply(preprocess_text)
#     for model_name in models_prompts:
#         logger.info(f"Generating embeddings using {model_name}")
#         model = SentenceTransformer(model_name, device=device)
#         embedding_col = f"{doc_col}_{model_name}"
#         df = generate_st_embeddings(
#             df=df,
#             doc_col=doc_col,
#             embedding_col=embedding_col,
#             model=model,
#             prompt=models_prompts[model_name].get("passage"),
#         )

#         del model
#         gc.collect()
#         if device.lower().startswith("cuda"):
#             torch.cuda.empty_cache()
#             torch.cuda.ipc_collect()
#     return df


# if __name__ == "__main__":
#     seed = 42
#     seed_everything(seed)

#     doc_col = "text"
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     dataset_path = Path("data") / "prepared" / "sankt-peterburg.csv"
#     dataset_name = dataset_path.stem
#     embeddings_path = (
#         Path("data") / "embedding" / f"st_embeddings_{dataset_name}.parquet"
#     )

#     if embeddings_path.exists():
#         logger.info(f"Loading embeddings from {str(embeddings_path)}")
#         df = pd.read_parquet(embeddings_path)
#     else:
#         logger.info(f"Existing {str(embeddings_path)} not found, using raw")
#         df = pd.read_csv(dataset_path)

#     df = generate_st_embeddings_models(df, doc_col, MODELS_PROMPTS)

#     embeddings_path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_parquet(embeddings_path, index=False)
