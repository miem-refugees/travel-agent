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

|     | experiment                                                                                               |    map@1 |    map@3 |    map@5 |   map@10 |   map@20 | benchmark_duration_sec | embedding_dim | num_params |
| --: | :------------------------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | -------: | ---------------------: | :------------ | :--------- |
|   0 | cointegrated/rubert-tiny2                                                                                |  0.12766 | 0.184397 | 0.173257 | 0.171266 | 0.158111 |                3.73813 | 312           | 29M        |
|   1 | DeepPavlov/rubert-base-cased-sentence                                                                    | 0.212766 |  0.27305 | 0.293735 | 0.281967 | 0.248308 |                3.13167 | 768           | 178M       |
|   2 | ai-forever/sbert_large_nlu_ru                                                                            | 0.191489 | 0.271277 | 0.274941 | 0.272615 | 0.243741 |                3.65721 | 1024          | 427M       |
|   3 | ai-forever/sbert_large_mt_nlu_ru                                                                         | 0.276596 |  0.35461 | 0.345952 | 0.315153 |  0.29529 |                3.89075 | 1024          | 427M       |
|   4 | sentence-transformers/distiluse-base-multilingual-cased-v1                                               | 0.276596 | 0.390071 | 0.399645 | 0.365621 | 0.344047 |                3.53789 | 512           | 135M       |
|   5 | sentence-transformers/distiluse-base-multilingual-cased-v2                                               | 0.212766 | 0.280142 | 0.285668 | 0.287939 | 0.269571 |                3.29144 | 512           | 135M       |
|   6 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2                                              | 0.382979 | 0.462766 | 0.461436 | 0.432146 | 0.398064 |                3.48334 | 384           | 118M       |
|   7 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2                                              | 0.361702 |  0.41844 | 0.392051 | 0.377196 | 0.361128 |                4.32264 | 768           | 278M       |
|   8 | intfloat/multilingual-e5-large                                                                           | 0.404255 | 0.496454 | 0.489569 | 0.466316 | 0.446634 |                4.94953 | 1024          | 560M       |
|   9 | intfloat/multilingual-e5-base                                                                            | 0.361702 | 0.480496 | 0.505822 | 0.475523 | 0.440102 |                 4.7077 | 768           | 278M       |
|  10 | intfloat/multilingual-e5-small                                                                           | 0.510638 | 0.567376 | 0.557772 | 0.512455 | 0.467331 |                4.08593 | 384           | 118M       |
|  11 | ai-forever/ru-en-RoSBERTa                                                                                | 0.382979 | 0.471631 | 0.486348 | 0.476196 | 0.443461 |                5.39178 | 1024          | 405M       |
|  12 | sergeyzh/BERTA                                                                                           | 0.404255 |  0.51773 | 0.519208 | 0.507564 | 0.478473 |                3.00399 | 768           | 128M       |
|  13 | Qdrant/bm25                                                                                              | 0.404255 | 0.480496 | 0.483363 | 0.477815 | 0.446986 |               0.153049 | -             | -          |
|  14 | jinaai/jina-colbert-v2                                                                                   | 0.425532 | 0.507092 | 0.522872 | 0.490707 | 0.459506 |                18.8804 | 128           | 559M       |
|  15 | cointegrated/rubert-tiny2+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                                   | 0.425532 | 0.547872 | 0.548641 | 0.513065 | 0.487613 |                15.5371 | -             | -          |
|  16 | DeepPavlov/rubert-base-cased-sentence+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                       | 0.531915 | 0.617021 | 0.591608 | 0.547811 |  0.50214 |                16.0887 | -             | -          |
|  17 | ai-forever/sbert_large_nlu_ru+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                               | 0.446809 | 0.572695 | 0.571749 | 0.526894 | 0.491882 |                18.7095 | -             | -          |
|  18 | ai-forever/sbert_large_mt_nlu_ru+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                            | 0.446809 | 0.592199 | 0.593322 | 0.538729 | 0.490819 |                18.8035 | -             | -          |
|  19 | sentence-transformers/distiluse-base-multilingual-cased-v1+Qdrant/bm25_reranking_jinaai/jina-colbert-v2  | 0.468085 | 0.539007 | 0.539509 | 0.512756 | 0.476714 |                12.4433 | -             | -          |
|  20 | sentence-transformers/distiluse-base-multilingual-cased-v2+Qdrant/bm25_reranking_jinaai/jina-colbert-v2  | 0.489362 | 0.585106 |  0.58357 |  0.53676 | 0.501649 |                16.7634 | -             | -          |
|  21 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2+Qdrant/bm25_reranking_jinaai/jina-colbert-v2 | 0.468085 | 0.553191 | 0.560343 | 0.523526 | 0.486309 |                14.2492 | -             | -          |
|  22 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2+Qdrant/bm25_reranking_jinaai/jina-colbert-v2 | 0.446809 | 0.583333 |  0.56711 | 0.525376 | 0.496127 |                14.4129 | -             | -          |
|  23 | intfloat/multilingual-e5-large+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                              | 0.425532 | 0.533688 |  0.52958 | 0.517469 | 0.485911 |                15.5302 | -             | -          |
|  24 | intfloat/multilingual-e5-base+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                               | 0.446809 | 0.551418 | 0.543824 | 0.516752 | 0.478803 |                14.8345 | -             | -          |
|  25 | intfloat/multilingual-e5-small+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                              | 0.382979 | 0.515957 | 0.528517 | 0.506841 | 0.469442 |                15.6836 | -             | -          |
|  26 | ai-forever/ru-en-RoSBERTa+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                                   | 0.446809 | 0.562057 | 0.558688 | 0.529818 | 0.486212 |                18.3562 | -             | -          |
|  27 | sergeyzh/BERTA+Qdrant/bm25_reranking_jinaai/jina-colbert-v2                                              | 0.510638 | 0.567376 | 0.556767 | 0.516262 | 0.473872 |                 15.894 | -             | -          |
|  28 | multi_stage_1000_Qdrant/bm25_top_k_cointegrated/rubert-tiny2                                             | 0.170213 | 0.303191 |  0.31182 |  0.31423 | 0.287678 |                3.00228 | -             | -          |
|  29 | multi_stage_1000_Qdrant/bm25_top_k_DeepPavlov/rubert-base-cased-sentence                                 | 0.255319 | 0.365248 | 0.371749 | 0.360414 | 0.331393 |                2.68737 | -             | -          |
|  30 | multi_stage_1000_Qdrant/bm25_top_k_ai-forever/sbert_large_nlu_ru                                         | 0.234043 | 0.329787 | 0.337766 | 0.330232 | 0.324457 |                3.79304 | -             | -          |
|  31 | multi_stage_1000_Qdrant/bm25_top_k_ai-forever/sbert_large_mt_nlu_ru                                      | 0.361702 | 0.441489 |  0.45393 | 0.428261 |   0.4014 |                4.30295 | -             | -          |
|  32 | multi_stage_1000_Qdrant/bm25_top_k_sentence-transformers/distiluse-base-multilingual-cased-v1            | 0.446809 | 0.526596 | 0.498522 | 0.468791 | 0.436813 |                3.02652 | -             | -          |
|  33 | multi_stage_1000_Qdrant/bm25_top_k_sentence-transformers/distiluse-base-multilingual-cased-v2            | 0.340426 | 0.443262 | 0.443203 |   0.4183 | 0.403616 |                3.08709 | -             | -          |
|  34 | multi_stage_1000_Qdrant/bm25_top_k_sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2           | 0.446809 | 0.524823 | 0.501448 | 0.495835 | 0.458688 |                3.46417 | -             | -          |
|  35 | multi_stage_1000_Qdrant/bm25_top_k_sentence-transformers/paraphrase-multilingual-mpnet-base-v2           | 0.361702 | 0.453901 | 0.454019 | 0.451147 | 0.431752 |                3.86228 | -             | -          |
|  36 | multi_stage_1000_Qdrant/bm25_top_k_intfloat/multilingual-e5-large                                        | 0.404255 | 0.510638 | 0.515751 | 0.497098 | 0.472866 |                4.76906 | -             | -          |
|  37 | multi_stage_1000_Qdrant/bm25_top_k_intfloat/multilingual-e5-base                                         | 0.425532 | 0.526596 | 0.527394 | 0.505217 | 0.476713 |                4.29702 | -             | -          |
|  38 | multi_stage_1000_Qdrant/bm25_top_k_intfloat/multilingual-e5-small                                        | 0.489362 | 0.583333 | 0.588268 | 0.557439 | 0.509326 |                5.00286 | -             | -          |
|  39 | multi_stage_1000_Qdrant/bm25_top_k_ai-forever/ru-en-RoSBERTa                                             | 0.404255 | 0.494681 | 0.493174 | 0.487386 |  0.45671 |                5.42286 | -             | -          |
|  40 | multi_stage_1000_Qdrant/bm25_top_k_sergeyzh/BERTA                                                        | 0.382979 | 0.514184 | 0.529521 | 0.513801 | 0.483387 |                 3.4609 | -             | -          |
|  41 | hybrid_search_top_models                                                                                 | 0.382979 | 0.492908 |  0.51117 | 0.481102 | 0.481102 |                31.8698 | -             | -          |

### Retrieval Strategies

The project compares:

- **Dense Retrieval**: Using vector embeddings to find semantically similar content
- **Sparse Retrieval**: Using BM25 and SPLADE for keyword-based retrieval
- **Hybrid Retrieval**: Combining dense and sparse approaches
- **Re-ranking**: Using cross-encoders or LLM-based re-ranking to refine results

## License

This project is licensed under the MIT License - see the LICENSE file for details.
