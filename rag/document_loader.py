from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredURLLoader


def load_documents(uploaded_files) -> list[Document]:
    """
    Extract text from Streamlit uploaded PDF/TXT files.
    PDF pages are separate Documents so page citations remain available.
    """
    documents: list[Document] = []

    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        extension = Path(filename).suffix.lower()
        file_bytes = uploaded_file.getvalue()

        if not file_bytes:
            raise ValueError(f"{filename} is empty.")

        if extension == ".pdf":
            try:
                reader = PdfReader(BytesIO(file_bytes))
                for page_number, page in enumerate(reader.pages, start=1):
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        documents.append(
                            Document(
                                page_content=page_text,
                                metadata={
                                    "source": filename,
                                    "page": page_number,
                                    "file_type": "pdf",
                                },
                            )
                        )
            except Exception as exc:
                raise ValueError(f"Could not read PDF '{filename}': {exc}") from exc

        elif extension == ".txt":
            try:
                text = file_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1")

            text = text.strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": None,
                            "file_type": "txt",
                        },
                    )
                )
        else:
            raise ValueError(
                f"Unsupported file type for '{filename}'. Upload PDF or TXT only."
            )

    return documents


def load_url_documents(urls: list[str]) -> list[Document]:
    """
    Fetch and preprocess article content from URLs using LangChain's
    UnstructuredURLLoader. URL documents do not have PDF page numbers.
    """
    clean_urls = [url.strip() for url in urls if url.strip()]
    if not clean_urls:
        return []

    for url in clean_urls:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}. URLs must start with http:// or https://.")

    try:
        loader = UnstructuredURLLoader(
            urls=clean_urls,
            continue_on_failure=False,
        )
        loaded_docs = loader.load()
    except Exception as exc:
        raise ValueError(
            "Could not load one or more URLs. Check the URLs and your internet "
            f"connection. Loader details: {exc}"
        ) from exc

    documents = []
    for doc in loaded_docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        source = doc.metadata.get("source", "URL source")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": None,
                    "file_type": "url",
                },
            )
        )
    return documents
