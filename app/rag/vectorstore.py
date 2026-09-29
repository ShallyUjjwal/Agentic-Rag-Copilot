import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None


EMBEDDING_DIMENSIONS = {
    "all-minilm-l6-v2": 384,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    """
    Return the vector dimension for the configured embedding model.

    Input:
        model_name: Optional embedding model name.

    Returns:
        int: Embedding vector dimension.
    """

    name = (model_name or settings.embedding_model or "").strip()

    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()

    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]

    if "all-minilm" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model "
        f"'{model_name or settings.embedding_model}' for Pinecone."
    )


def get_embeddings():
    """
    Create and cache the Hugging Face embedding model.

    Returns:
        HuggingFaceEmbeddings: Embedding model.
    """

    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

    return _embeddings


def ensure_index():
    """
    Connect to Pinecone and ensure the required index exists
    with the correct embedding dimension.

    Returns:
        Pinecone Index object.
    """

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(
        api_key=settings.pinecone_api_key
    )

    existing_indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    # If the index already exists, verify its dimension.
    if settings.pinecone_index_name in existing_indexes:

        index_info = pc.describe_index(
            settings.pinecone_index_name
        )

        current_dimension = getattr(
            index_info,
            "dimension",
            None,
        )

        if current_dimension is None and isinstance(
            index_info,
            dict,
        ):
            current_dimension = index_info.get(
                "dimension"
            )

        if (
            current_dimension is not None
            and current_dimension != desired_dimension
        ):
            raise RuntimeError(
                f"Pinecone index '{settings.pinecone_index_name}' "
                f"has dimension {current_dimension}, but the "
                f"configured embedding model produces "
                f"{desired_dimension}-dimensional vectors. "
                f"Use a matching index or recreate the index."
            )

    # Create index if it doesn't exist.
    if settings.pinecone_index_name not in [
        index["name"]
        for index in pc.list_indexes()
    ]:

        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        # Wait until Pinecone finishes creating the index.
        while not pc.describe_index(
            settings.pinecone_index_name
        ).status["ready"]:
            time.sleep(1)

    return pc.Index(
        settings.pinecone_index_name
    )


def get_vectorstore():
    """
    Create and cache the Pinecone vector store.

    Returns:
        PineconeVectorStore: LangChain Pinecone vector store.
    """

    global _vectorstore

    if _vectorstore is None:

        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    """
    Return a Pinecone retriever.

    Returns:
        VectorStoreRetriever: Retriever configured with top_k.
    """

    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )


def add_documents(chunks):
    """
    Add document chunks to Pinecone.

    Input:
        chunks: List of LangChain Document objects.

    Returns:
        List of IDs created by Pinecone.
    """

    store = get_vectorstore()

    return store.add_documents(chunks)