
qdrant_client.create_collection(
    collection_name="movies",
    vectors_config=models.VectorParams(
        size=128,  # size of each vector produced by ColBERT
        distance=models.Distance.COSINE,  # similarity metric between each vector
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM  # similarity metric between multivectors (matrices)
        ),
    ),
)


description_documents = [models.Document(text=description, model=model_name) for description in descriptions]
qdrant_client.upload_points(
    collection_name="movies",
    points=[
        models.PointStruct(id=idx, payload=metadata[idx], vector=description_document)
        for idx, description_document in enumerate(description_documents)
    ],
)


qdrant_client.query_points(
    collection_name="movies",
    query=list(embedding_model.query_embed("A movie for kids with fantasy elements and wonders"))[
        0
    ],  # converting generator object into numpy.ndarray
    limit=1,  # How many closest to the query movies we would like to get
    # with_vectors=True, #If this option is used, vectors will also be returned
    with_payload=True,  # So metadata is provided in the output
)
