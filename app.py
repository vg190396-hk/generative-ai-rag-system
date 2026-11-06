import streamlit as st

from config import APP_TITLE, APP_ICON, MAX_FILE_SIZE_MB
from rag.document_loader import load_documents, load_url_documents
from rag.text_splitter import split_documents
from rag.vector_store import create_vector_store
from rag.qa_chain import answer_question
from utils.helpers import format_source_label, get_unique_sources


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 16px;
        background: linear-gradient(120deg, #172554 0%, #1d4ed8 60%, #0f766e 100%);
        color: white; margin-bottom: 1.3rem;
    }
    .hero h1 {color: white; margin: 0 0 .35rem 0; font-size: 2rem;}
    .hero p {color: #dbeafe; margin: 0; font-size: 1rem;}
    .small-muted {color: #64748b; font-size: .88rem;}
    .source-card {
        border: 1px solid rgba(128,128,128,.3); border-radius: 10px;
        padding: .85rem 1rem; margin: .5rem 0;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        padding: 12px; border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown(
    """
    <div class="hero">
      <h1>Document Intelligence</h1>
      <p>Ask questions about your PDFs and text files. Get context-grounded answers with source references.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Knowledge Base")
    st.write("Upload documents to create a searchable knowledge base.")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help=f"Maximum size: {MAX_FILE_SIZE_MB} MB per file.",
    )
    url_text = st.text_area(
        "Article URLs (optional)",
        placeholder="Paste one http:// or https:// URL per line",
        height=90,
    )
    url_list = [line.strip() for line in url_text.splitlines() if line.strip()]
    has_input = bool(uploaded_files) or bool(url_list)

    if uploaded_files:
        oversized = [
            f.name for f in uploaded_files
            if f.size > MAX_FILE_SIZE_MB * 1024 * 1024
        ]
        if oversized:
            st.error(
                f"These files exceed {MAX_FILE_SIZE_MB} MB: "
                + ", ".join(oversized)
            )

    process_clicked = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
        disabled=not has_input,
    )

    if process_clicked:
        try:
            if any(f.size > MAX_FILE_SIZE_MB * 1024 * 1024 for f in uploaded_files):
                st.error(f"Please upload files smaller than {MAX_FILE_SIZE_MB} MB each.")
            else:
                with st.spinner("Extracting text, creating chunks and building the FAISS index..."):
                    documents = []
                    if uploaded_files:
                        documents.extend(load_documents(uploaded_files))
                    if url_list:
                        documents.extend(load_url_documents(url_list))
                    if not documents:
                        st.error("No readable text was found in the uploaded files.")
                    else:
                        chunks = split_documents(documents)
                        if not chunks:
                            st.error("The uploaded files did not produce any text chunks.")
                        else:
                            st.session_state.vector_store = create_vector_store(chunks)
                            st.session_state.processed_files = sorted(
                                {doc.metadata.get("source", "Unknown") for doc in documents}
                            )
                            st.session_state.chunk_count = len(chunks)
                            st.session_state.chat_history = []
                            st.success(
                                f"Knowledge base ready: {len(documents)} page/document "
                                f"records split into {len(chunks)} chunks."
                            )
        except EnvironmentError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Document processing failed: {exc}")

    st.divider()
    st.subheader("Current index")
    if st.session_state.vector_store is not None:
        st.metric("Text chunks indexed", st.session_state.chunk_count)
        for filename in st.session_state.processed_files:
            st.caption(f"• {filename}")
        if st.button("Clear knowledge base", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.processed_files = []
            st.session_state.chunk_count = 0
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.caption("No documents processed yet.")

left, right = st.columns([1.55, 1], gap="large")

with left:
    st.subheader("Ask your documents")
    if st.session_state.vector_store is None:
        st.info("Upload PDF/TXT files or paste article URLs in the sidebar, then select **Process documents**.")
    else:
        question = st.text_area(
            "Your question",
            placeholder="Example: What are the main findings discussed in the document?",
            height=100,
            key="question_input",
        )
        ask_clicked = st.button("Generate answer", type="primary")
        if ask_clicked:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner("Searching your documents and generating an answer..."):
                        result = answer_question(st.session_state.vector_store, question.strip())
                    st.session_state.chat_history.insert(0, result)
                except EnvironmentError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Could not generate an answer: {exc}")

        if st.session_state.chat_history:
            for index, item in enumerate(st.session_state.chat_history):
                st.markdown("#### Answer" if index == 0 else "#### Previous answer")
                st.markdown(item["answer"])
                with st.expander(f"Sources used ({len(item['sources'])})", expanded=(index == 0)):
                    for source in item["sources"]:
                        st.markdown(
                            f"""
                            <div class="source-card">
                              <strong>{format_source_label(source)}</strong><br>
                              <span class="small-muted">{source.get("excerpt", "")}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

with right:
    st.subheader("How it works")
    st.markdown(
        """
        1. **Upload** PDF/TXT documents or provide article URLs.
        2. **Extract** text and preserve source/page metadata.
        3. **Chunk** long text into smaller overlapping passages.
        4. **Embed** each passage using OpenAI Embeddings.
        5. **Index** the vectors in FAISS.
        6. **Retrieve** relevant passages for each question.
        7. **Generate** an answer using the retrieved context.
        """
    )
    st.divider()
    st.subheader("Important")
    st.caption(
        "Answers are generated from retrieved document passages. "
        "Check the displayed source references before relying on an answer. "
        "Scanned/image-only PDFs may require OCR, which this project does not include. "
        "URL loading depends on the source website allowing automated access."
    )
