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

| Model | MAP@1 | MAP@3 | MAP@5 | MAP@10 | MAP@20 |
|-------|-------|-------|-------|--------|--------|
| DeepPavlov/rubert-base-cased-sentence | 0.079 | 0.149 | 0.163 | 0.150 | 0.155 |
| distiluse-base-multilingual-cased-v1 | 0.132 | 0.202 | 0.217 | 0.224 | 0.207 |
| distiluse-base-multilingual-cased-v2 | 0.158 | 0.228 | 0.239 | 0.248 | 0.239 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.263 | 0.300 | 0.319 | 0.326 | 0.300 |
| paraphrase-multilingual-mpnet-base-v2 | 0.158 | 0.309 | 0.324 | 0.324 | 0.311 |

#### Visual Comparison

```
MAP@10 Performance Comparison
----------------------------------
paraphrase-multilingual-MiniLM-L12-v2  ██████████████████████████████  0.326
paraphrase-multilingual-mpnet-base-v2  ██████████████████████████████  0.324
distiluse-base-multilingual-cased-v2   █████████████████████████       0.248
distiluse-base-multilingual-cased-v1   ███████████████████             0.224
DeepPavlov/rubert-base-cased-sentence  ███████████████                 0.150
```

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
