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
```

### Data Preparation

The project uses DVC to manage the data pipeline:

```bash
# single command for all pipeline:
dvc repro
```

### Embedding Benchmark Results

We've benchmarked several multilingual embedding models on our dataset to measure retrieval performance. The key metric we use is Mean Average Precision (MAP) at different k values.

#### MAP@k Results

|    | experiment                                                                                               |    map@1 |    map@3 |    map@5 |   map@10 |   map@20 |   benchmark_duration_sec | embedding_dim   | num_params   |
|---:|:---------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|---------:|---------:|-------------------------:|:----------------|:-------------|
|  0 | cointegrated/rubert-tiny2                                                                                | 0.12766  | 0.18617  | 0.173848 | 0.168606 | 0.157489 |                 3.98638  | 312             | 29M          |
|  1 | DeepPavlov/rubert-base-cased-sentence                                                                    | 0.212766 | 0.26773  | 0.284515 | 0.272984 | 0.244167 |                 2.73323  | 768             | 178M         |
|  2 | ai-forever/sbert_large_nlu_ru                                                                            | 0.212766 | 0.271277 | 0.274232 | 0.269268 | 0.240998 |                 3.70873  | 1024            | 427M         |
|  3 | ai-forever/sbert_large_mt_nlu_ru                                                                         | 0.276596 | 0.356383 | 0.345952 | 0.318414 | 0.298798 |                 3.64512  | 1024            | 427M         |
|  4 | sentence-transformers/distiluse-base-multilingual-cased-v1                                               | 0.297872 | 0.397163 | 0.404728 | 0.367698 | 0.344475 |                 3.07567  | 512             | 135M         |
|  5 | sentence-transformers/distiluse-base-multilingual-cased-v2                                               | 0.191489 | 0.285461 | 0.286909 | 0.288297 | 0.272745 |                 3.12951  | 512             | 135M         |
|  6 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2                                              | 0.361702 | 0.453901 | 0.446188 | 0.419447 | 0.386504 |                 3.38915  | 384             | 118M         |
|  7 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2                                              | 0.382979 | 0.420213 | 0.394178 | 0.377136 | 0.361083 |                 3.73736  | 768             | 278M         |
|  8 | intfloat/multilingual-e5-large                                                                           | 0.425532 | 0.5      | 0.49276  | 0.468168 | 0.450785 |                 4.88641  | 1024            | 560M         |
|  9 | intfloat/multilingual-e5-base                                                                            | 0.404255 | 0.496454 | 0.513505 | 0.477439 | 0.443799 |                 4.3108   | 768             | 278M         |
| 10 | intfloat/multilingual-e5-small                                                                           | 0.425532 | 0.512411 | 0.514953 | 0.476494 | 0.439935 |                 4.35427  | 384             | 118M         |
| 11 | ai-forever/ru-en-RoSBERTa                                                                                | 0.361702 | 0.452128 | 0.472518 | 0.471765 | 0.446313 |                 5.38333  | 1024            | 405M         |
| 12 | sergeyzh/BERTA                                                                                           | 0.425532 | 0.507092 | 0.5151   | 0.500048 | 0.473414 |                 3.0437   | 768             | 128M         |
| 13 | Qdrant/bm25                                                                                              | 0.404255 | 0.496454 | 0.483481 | 0.475132 | 0.446098 |                 0.140611 | -               | -            |
| 14 | jinaai/jina-colbert-v2                                                                                   | 0.425532 | 0.51773  | 0.52896  | 0.491759 | 0.461816 |                12.2201   | 128             | 559M         |
| 15 | cointegrated/rubert-tiny2+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                                   | 0.425532 | 0.542553 | 0.547754 | 0.514325 | 0.485871 |                13.714    | -               | -            |
| 16 | DeepPavlov/rubert-base-cased-sentence+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                       | 0.510638 | 0.599291 | 0.582388 | 0.540685 | 0.497808 |                14.607    | -               | -            |
| 17 | ai-forever/sbert_large_nlu_ru+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                               | 0.489362 | 0.601064 | 0.58753  | 0.539977 | 0.496107 |                13.0974   | -               | -            |
| 18 | ai-forever/sbert_large_mt_nlu_ru+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                            | 0.425532 | 0.585106 | 0.592435 | 0.540853 | 0.490978 |                12.9831   | -               | -            |
| 19 | sentence-transformers/distiluse-base-multilingual-cased-v1+Qdrant/bm25_reranking_jinaai/jina-colbert-v2  | 0.446809 | 0.533688 | 0.544297 | 0.513253 | 0.477142 |                12.1212   | -               | -            |
| 20 | sentence-transformers/distiluse-base-multilingual-cased-v2+Qdrant/bm25_reranking_jinaai/jina-colbert-v2  | 0.446809 | 0.556738 | 0.555201 | 0.519752 | 0.491958 |                12.2971   | -               | -            |
| 21 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2+Qdrant/bm25_reranking_jinaai/jina-colbert-v2 | 0.425532 | 0.54078  | 0.555171 | 0.518528 | 0.48477  |                13.2642   | -               | -            |
| 22 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2+Qdrant/bm25_reranking_jinaai/jina-colbert-v2 | 0.446809 | 0.583333 | 0.56977  | 0.525699 | 0.494034 |                13.448    | -               | -            |
| 23 | intfloat/multilingual-e5-large+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                              | 0.446809 | 0.528369 | 0.520715 | 0.510914 | 0.479422 |                15.2014   | -               | -            |
| 24 | intfloat/multilingual-e5-base+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                               | 0.425532 | 0.556738 | 0.553251 | 0.517662 | 0.478473 |                13.8399   | -               | -            |
| 25 | intfloat/multilingual-e5-small+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                              | 0.446809 | 0.537234 | 0.547252 | 0.510803 | 0.474394 |                13.8324   | -               | -            |
| 26 | ai-forever/ru-en-RoSBERTa+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                                   | 0.446809 | 0.569149 | 0.56513  | 0.527441 | 0.486501 |                16.7176   | -               | -            |
| 27 | sergeyzh/BERTA+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                                              | 0.446809 | 0.533688 | 0.531058 | 0.505956 | 0.467606 |                13.7417   | -               | -            |
| 28 | multi_stage_1000_Qdrant/bm25_top_k_cointegrated/rubert-tiny2                                             | 0.170213 | 0.297872 | 0.306619 | 0.308356 | 0.284675 |                 2.9336   | -               | -            |
| 29 | multi_stage_1000_Qdrant/bm25_top_k_DeepPavlov/rubert-base-cased-sentence                                 | 0.234043 | 0.35461  | 0.356856 | 0.359375 | 0.330505 |                 2.79     | -               | -            |
| 30 | multi_stage_1000_Qdrant/bm25_top_k_ai-forever/sbert_large_nlu_ru                                         | 0.234043 | 0.329787 | 0.334929 | 0.329854 | 0.324031 |                 4.31773  | -               | -            |
| 31 | multi_stage_1000_Qdrant/bm25_top_k_ai-forever/sbert_large_mt_nlu_ru                                      | 0.382979 | 0.437943 | 0.452985 | 0.431871 | 0.40207  |                 3.85614  | -               | -            |
| 32 | multi_stage_1000_Qdrant/bm25_top_k_sentence-transformers/distiluse-base-multilingual-cased-v1            | 0.446809 | 0.514184 | 0.500739 | 0.471549 | 0.437482 |                 3.61652  | -               | -            |
| 33 | multi_stage_1000_Qdrant/bm25_top_k_sentence-transformers/distiluse-base-multilingual-cased-v2            | 0.361702 | 0.45922  | 0.456501 | 0.418857 | 0.404294 |                 3.06401  | -               | -            |
| 34 | multi_stage_1000_Qdrant/bm25_top_k_sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2           | 0.489362 | 0.542553 | 0.514923 | 0.503647 | 0.465368 |                 3.8264   | -               | -            |
| 35 | multi_stage_1000_Qdrant/bm25_top_k_sentence-transformers/paraphrase-multilingual-mpnet-base-v2           | 0.382979 | 0.45922  | 0.460284 | 0.456012 | 0.433776 |                 4.18046  | -               | -            |
| 36 | multi_stage_1000_Qdrant/bm25_top_k_intfloat/multilingual-e5-large                                        | 0.404255 | 0.501773 | 0.505467 | 0.492915 | 0.471671 |                 5.09563  | -               | -            |
| 37 | multi_stage_1000_Qdrant/bm25_top_k_intfloat/multilingual-e5-base                                         | 0.425532 | 0.54078  | 0.532417 | 0.499876 | 0.468945 |                 4.22179  | -               | -            |
| 38 | multi_stage_1000_Qdrant/bm25_top_k_intfloat/multilingual-e5-small                                        | 0.553191 | 0.615248 | 0.601241 | 0.56224  | 0.509358 |                 4.52473  | -               | -            |
| 39 | multi_stage_1000_Qdrant/bm25_top_k_ai-forever/ru-en-RoSBERTa                                             | 0.425532 | 0.501773 | 0.504285 | 0.491842 | 0.460174 |                 5.51846  | -               | -            |
| 40 | multi_stage_1000_Qdrant/bm25_top_k_sergeyzh/BERTA                                                        | 0.404255 | 0.514184 | 0.535786 | 0.520057 | 0.487631 |                 2.99704  | -               | -            |
| 41 | hybrid_search_top_models                                                                                 | 0.425532 | 0.515957 | 0.515957 | 0.482628 | 0.482628 |                31.9724   | -               | -            |
| 42 | hybrid_search_top_models_2                                                                               | 0.425532 | 0.512411 | 0.523522 | 0.497036 | 0.46367  |                10.5266   | -               | -            |
| 43 | hybrid_search_top_models_3                                                                               | 0.468085 | 0.546099 | 0.525798 | 0.493025 | 0.455922 |                19.1039   | -               | -            |

### Retrieval Strategies

The project compares:

- **Dense Retrieval**: Using vector embeddings to find semantically similar content
- **Sparse Retrieval**: Using BM25 and SPLADE for keyword-based retrieval
- **Hybrid Retrieval**: Combining dense and sparse approaches
- **Re-ranking**: Using cross-encoders or LLM-based re-ranking to refine results

## License

This project is licensed under the MIT License - see the LICENSE file for details.
