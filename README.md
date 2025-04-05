# **Travel Agent 🕵🏾**  

**Enhancing Trip Planning with Retrieval-Augmented Generation (RAG)**  
**Dataset:** [Yandex Geo Reviews 2023](https://github.com/yandex/geo-reviews-dataset-2023)

## ✅ **Чеклист: Retrieval в RAG**

### 1. 📌 **Анализ текущей dense-модели**
- [ ] Обосновать выбор `intfloat/multilingual-e5-large`
  - Мультиязычность
  - Качество на retrieval задачах (MTEB, BEIR)
  - Баланс производительности / размера
- [ ] Сравнить с другими dense-моделями:
  - [ ] Меньше: `intfloat/multilingual-e5-small`, `bge-small-en`, `all-MiniLM-L6-v2`
  - [ ] Больше: `bge-large-en`, `e5-mistral-7b`
  - [ ] Такие же: `bge-base-en-v1.5`, `all-mpnet-base-v2`
- [ ] Визуализировать embeddings (UMAP/t-SNE) на примере


### 2. 🧵 **Сравнение с sparse retrieval**
- [ ] Использовать BM25 (из `faiss`, `rank_bm25`, `elasticsearch` или `whoosh`)
- [ ] Попробовать SPLADE (через `BeIR` или `pyserini`)
- [ ] Провести сравнение качества (например, по nDCG@10, Recall@k)


### 3. 🧪 **Построение гибридного поиска**
- [ ] Скомбинировать dense + sparse (например, BM25 + E5)
  - [ ] Normalize и объединить score'ы (взвешенное среднее)
  - [ ] Протестировать разные веса
- [ ] Построить гибридный retrieval pipeline


### 4. 🏗️ **Late interaction / reranking**
- [ ] Попробовать ColBERT (через `colbert-ai/colbert`)
  - [ ] Замерить скорость и качество
  - [ ] Возможность использовать как reranker
- [ ] Попробовать Cross-encoders:
  - [ ] `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - [ ] `cross-encoder/ms-marco-TinyBERT-L-2-v2`
- [ ] Попробовать LLM-based reranker:
  - [ ] Простой prompt на OpenAI API или LLaMA 2
  - [ ] Написать scoring функцию


### 5. ⚙️ **Скрипт-бенчмаркер**
- [ ] Python-скрипт `benchmark.py`, который:
  - [ ] Принимает параметры: имя модели, тип retrieval (`dense`, `sparse`, `hybrid`)
  - [ ] Загружает коллекцию (в формате JSON/CSV)
  - [ ] Генерирует эмбеддинги / индексы
  - [ ] Прогоняет поиск по queries
  - [ ] Выдает метрики: Recall@k, MRR, nDCG, latency
  - [ ] Сохраняет результаты (в JSON/CSV + графики)
- [ ] Поддержка разных источников: HuggingFace, FAISS, pyserini


### 6. 📊 **Результаты и отчет**
- [ ] Построить таблицу сравнения (по метрикам, скорости)
- [ ] Вывести лучшие комбинации retrieval + reranking
- [ ] Сделать выводы по trade-offs (качество vs скорость)



## 🛠️ **Tech Stack**

- **Python** 3.13  
- **DVC** - Data Version Control: https://dvc.org/doc/install 
- **Qdrant** – Vector database for efficient dense retrieval: https://qdrant.tech
- **FastEmbed** – Lightweight embedding generation with HuggingFace models
- **SmolAgents** – Compact agent framework for orchestrating RAG flows

## **Dev install**

Установите **DVC**, **AWS CLI** и настройте для работы с яндекс облаком: https://yandex.cloud/ru/docs/storage/tools/aws-cli 
