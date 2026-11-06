from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    validate_configuration,
)


def create_embeddings() -> OpenAIEmbeddings:
    validate_configuration()
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )


def create_vector_store(chunks) -> FAISS:
    """Create a FAISS vector index from LangChain Documents."""
    if not chunks:
        raise ValueError("Cannot create a vector store because there are no chunks.")

    embeddings = create_embeddings()
    return FAISS.from_documents(chunks, embeddings)
