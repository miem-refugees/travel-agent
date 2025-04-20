# 🚀 Getting started

## Overview

Travel Agent is a Trip Planning Assistant powered by Retrieval-Augmented Generation (RAG) technology. The system utilizes the [Yandex Geo Reviews 2023](https://github.com/yandex/geo-reviews-dataset-2023) dataset to provide location-aware recommendations for places, businesses, and services.

## Project Structure

- **Data Pipeline**: Prepares and analyzes the Yandex Geo Reviews dataset
- **Embedding Scripts**: Implements comparsion of different embedding models for Russian language

## Getting Started

### Prerequisites

- Python 3.13
- [DVC](https://dvc.org/doc/install) for data version control
- [AWS CLI](https://yandex.cloud/ru/docs/storage/tools/aws-cli) configured for Yandex Cloud

### Installation

```bash
# Clone the repository
git clone https://github.com/miem-refugees/travel-agent.git
cd travel-agent

# Create a virtual environment with uv
uv venv

# Install dependencies
uv sync

# Install dev dependencies
uv sync --dev

# Install docs dependencies
uv sync --group docs
```

### Data Preparation

The project uses DVC to manage the data pipeline:

```bash
# Download and prepare the dataset
dvc repro prepare

# Analyze the dataset
dvc repro analyze

# Generate evaluation questions and datasets
dvc repro generate_moscow_rag_questions
dvc repro generate_saint_petersburg_rag_questions
dvc repro generate_norm_rubrics_questions
```

### Embedding Benchmark Results

We've benchmarked several multilingual embedding models on our dataset to measure retrieval performance. The key metric we use is Mean Average Precision (MAP) at different k values.

#### MAP@k Results

| model                                                       | map@1 | map@3 | map@5 | map@10 | map@20 | embedding_duration_sec | benchmark_duration_sec | total_duration_sec |
| :---------------------------------------------------------- | ----: | ----: | ----: | -----: | -----: | ---------------------: | ---------------------: | -----------------: |
| cointegrated/rubert-tiny2                                   |  0.19 |  0.21 |  0.19 |   0.18 |   0.16 |                  16.09 |                  11.24 |              27.32 |
| DeepPavlov/rubert-base-cased-sentence                       |  0.19 |  0.26 |  0.29 |   0.29 |   0.26 |                 207.21 |                  24.82 |             232.03 |
| ai-forever/sbert_large_nlu_ru                               |  0.23 |  0.26 |  0.27 |   0.27 |   0.24 |                 674.55 |                  32.81 |             707.37 |
| ai-forever/sbert_large_mt_nlu_ru                            |   0.3 |  0.36 |  0.35 |   0.34 |   0.31 |                 675.73 |                  33.34 |             709.07 |
| sentence-transformers/distiluse-base-multilingual-cased-v1  |   0.3 |  0.37 |   0.4 |   0.36 |   0.35 |                 123.22 |                   17.4 |             140.62 |
| sentence-transformers/distiluse-base-multilingual-cased-v2  |  0.23 |  0.31 |   0.3 |    0.3 |   0.28 |                 124.14 |                  17.38 |             141.52 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |  0.38 |  0.45 |  0.46 |   0.42 |   0.39 |                   68.9 |                  14.62 |              83.52 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |  0.32 |   0.4 |  0.38 |   0.37 |   0.36 |                 220.55 |                  25.11 |             245.67 |
| intfloat/multilingual-e5-large                              |  0.45 |   0.5 |  0.52 |   0.49 |   0.46 |                    857 |                  33.62 |             890.62 |
| intfloat/multilingual-e5-base                               |  0.45 |  0.52 |   0.5 |   0.49 |   0.45 |                 251.67 |                  24.92 |             276.59 |
| intfloat/multilingual-e5-small                              |  0.45 |  0.54 |  0.52 |   0.49 |   0.45 |                  80.22 |                  14.37 |              94.59 |
| ai-forever/ru-en-RoSBERTa                                   |  0.43 |   0.5 |   0.5 |   0.47 |   0.45 |                  778.2 |                  33.57 |             811.77 |
| sergeyzh/BERTA                                              |  0.45 |  0.53 |  0.54 |   0.52 |   0.48 |                  244.3 |                  25.45 |             269.75 |
| tfidf                                                       |     0 |     0 |     0 |      0 |      0 |                   0.31 |                1067.18 |             1067.5 |

#### Key Findings

- **Best Overall Performance**: The `paraphrase-multilingual-MiniLM-L12-v2` model demonstrates the highest MAP scores across most k values, making it our top performer.
- **Size-Performance Balance**: Despite having fewer parameters than mpnet-base, the MiniLM-L12 model shows slightly better performance, offering an excellent balance between model size and retrieval capability.
- **Russian Language Performance**: Specialized Russian BERT model from DeepPavlov performs worse than the multilingual models, possibly due to the specific nature of our review dataset and query patterns.

Based on these results, we've chosen to prioritize `paraphrase-multilingual-MiniLM-L12-v2` for our production implementation while exploring alternative and hybrid approaches to further improve performance.

### Retrieval Strategies

The project compares:

- **Dense Retrieval**: Using vector embeddings to find semantically similar content
- **Sparse Retrieval**: Using BM25 and SPLADE for keyword-based retrieval
- **Hybrid Retrieval**: Combining dense and sparse approaches
- **Re-ranking**: Using cross-encoders or LLM-based re-ranking to refine results

## Running Benchmarks

TODO @searayeah

## Usage

TODO :in progress:

## Contributing

See the [README.md](README.md) for the current development checklist and project roadmap. Please follow the project's code style using `pre-commit` hooks.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
