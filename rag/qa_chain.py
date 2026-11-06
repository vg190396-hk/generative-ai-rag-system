from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    RETRIEVAL_K,
    validate_configuration,
)


SYSTEM_PROMPT = """
You are a document question-answering assistant.
Answer the user's question using ONLY the context passages provided below.

Rules:
- Do not use outside knowledge to fill gaps.
- If the context does not contain the answer, clearly say:
  "I could not find enough information in the uploaded documents to answer this."
- Be clear and concise, and distinguish facts from reasonable interpretation.
- Add inline citations such as [Source 1] for claims supported by a passage.
- Never invent a source, page number, quotation, statistic, or citation.
"""


def answer_question(vector_store, question: str) -> dict:
    """Retrieve relevant chunks, ask the LLM, and return grounded source metadata."""
    validate_configuration()

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if vector_store is None:
        raise ValueError("Please upload and process documents first.")

    retrieved_docs = vector_store.similarity_search(
        question.strip(),
        k=RETRIEVAL_K,
    )

    if not retrieved_docs:
        return {
            "answer": "I could not find relevant information in the uploaded documents.",
            "sources": [],
        }

    context_parts = []
    sources = []

    for index, doc in enumerate(retrieved_docs, start=1):
        source_name = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page")
        page_label = f"Page {page}" if page is not None else "Page not available"
        excerpt = doc.page_content.strip()

        context_parts.append(
            f"[Source {index} | File: {source_name} | {page_label}]\n{excerpt}"
        )
        sources.append(
            {
                "source": source_name,
                "page": page,
                "excerpt": excerpt[:700] + ("..." if len(excerpt) > 700 else ""),
            }
        )

    context = "\n\n---\n\n".join(context_parts)
    human_message = f"""
Context passages:
{context}

User question:
{question}

Write the answer with citations referring to the source labels above.
"""

    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_message),
        ]
    )

    answer_text = response.content
    if isinstance(answer_text, list):
        answer_text = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in answer_text
        )

    return {
        "answer": str(answer_text).strip(),
        "sources": sources,
    }
