# RAG Implementation

Retrieval-Augmented Generation

## Overview

The RAG system combines semantic search capabilities with a language model to provide accurate and relevant information about travel destinations and establishments based on user reviews. The system uses vector embeddings to find semantically similar reviews and then uses a language model to generate natural language responses.

## Components

### 1. Embedding Model

The system uses the `intfloat/multilingual-e5-base` as most basic efficient model for generating embeddings. This model is particularly well-suited for multilingual text and provides high-quality semantic representations.

Key features:
- Supports multiple languages
- Efficient base model architecture
- Good performance on semantic similarity tasks

### 2. Vector Database

The system uses [Qdrant](https://qdrant.tech) as the vector database to store and retrieve embeddings. Each review is stored with the following metadata:
- Review text (vectorized)
- Name
- Rubrics
- Rating (1-5)
- Address

### 3. Search Tool

The `TravelReviewQueryTool` implements the core retrieval functionality with the following features:

- Semantic search using embeddings
- Filtering capabilities:
  - Minimum rating filter
  - Address/location filter
  - Category/rubric filter
- Configurable retrieval limit (default: 5 results)

### 4. Language Model Integration

The system supports multiple language models through different frameworks using ReAct agent pattern:
1. **SmolAgents**
2. **LangChain**

## Usage

### Basic Query Example

```python
# Initialize the search tool
review_search_tool = TravelReviewQueryTool(
    embed_model_name="intfloat/multilingual-e5-base",
    qdrant_client=qdrant_client,
    collection_name="moskva_intfloat_multilingual_e5_base"
)

# Example query
results = review_search_tool.forward(
    query="посоветуй хороший японский ресторан в Москве",
    min_rating=4
)
```

### Available Filters

1. **Query**: Natural language query for semantic search
2. **Min Rating**: Filter establishments by minimum rating (1-5)
3. **Address**: Filter by location (city or street)
4. **Rubrics**: Filter by establishment categories

## Evaluation

The system includes evaluation capabilities to measure performance:

1. **Relevancy Evaluation**
   - Uses Gemini model for evaluation
   - Measures if retrieved results are relevant to the query
   - Provides relevancy scores and explanations

2. **Hallucination Detection**
   - Evaluates if generated responses contain factual information
   - Uses predefined templates for consistency

## Performance Metrics

The system has been evaluated with the following metrics:
- Mean relevancy score: ~0.93
- High consistency in providing relevant results

## Best Practices

1. **Query Formulation**
   - Use natural language queries
   - Be specific about requirements
   - Include location when relevant

2. **Response Generation**
   - Don't repeat entire review text
   - Summarize key points
   - Focus on relevant information

3. **Error Handling**
   - Handle cases when no results are found
   - Provide helpful feedback to users
   - Use appropriate fallback strategies

## Future Improvements

1. **Model Enhancements**
   - Experiment with different embedding models
   - Fine-tune models on travel-specific data
   - Implement hybrid search strategies

2. **Feature Additions**
   - Add more sophisticated filtering options
   - Implement result ranking improvements
   - Add support for more languages

3. **Evaluation**
   - Expand evaluation metrics
   - Add more comprehensive testing
   - Implement automated quality checks
