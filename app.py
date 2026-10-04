"""Streamlit interface for local PDF retrieval and answer generation."""

import streamlit as st

from src.rag_project.chunker import chunk_pdf_pages
from src.rag_project.embeddings import embed_chunks, load_embedding_model
from src.rag_project.generator import (
    OLLAMA_MODEL,
    OllamaError,
    generate_grounded_answer,
)
from src.rag_project.pdf_loader import extract_pdf_pages
from src.rag_project.search import semantic_search


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load the model once and reuse it across Streamlit reruns."""
    return load_embedding_model()


st.set_page_config(
    page_title="Local PDF AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0f1420; color: #eef2ff; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #31265c 0%, #46327a 58%, #334c75 100%);
    }
    [data-testid="stSidebar"] * { color: #eef4ff; }
    [data-testid="stSidebar"] .stCaptionContainer { color: #d1c9e8; }
    .block-container { max-width: 1120px; padding-top: 2.2rem; }
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4 { color: #eef2ff; }
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        color: #aeb8cc;
    }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #4a2f7d 0%, #6741a5 55%, #168a8c 100%);
        color: white;
        box-shadow: 0 16px 38px rgba(74, 47, 125, 0.18);
        margin-bottom: 1.4rem;
    }
    .hero-kicker {
        color: #8ef0dc;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .hero h1 { font-size: 2.35rem; margin: 0; color: white; }
    .hero p { color: #eee8fa; font-size: 1.03rem; margin: 0.65rem 0 0; }
    .privacy-pill {
        display: inline-block;
        background: rgba(98, 231, 205, 0.14);
        color: #9af4df;
        border: 1px solid rgba(154, 244, 223, 0.35);
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        margin-top: 1rem;
        font-size: 0.78rem;
        font-weight: 650;
    }
    [data-testid="stMetric"] {
        background: #1a2232;
        border: 1px solid #343d52;
        border-radius: 15px;
        padding: 0.9rem 1rem;
        box-shadow: 0 4px 14px rgba(74, 47, 125, 0.06);
    }
    [data-testid="stMetricLabel"] p { color: #b7c0d8 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stFileUploader"] {
        background: #1a2232;
        border-radius: 15px;
    }
    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        border-radius: 11px;
        min-height: 2.8rem;
        font-weight: 700;
        background: #6741a5;
        color: white;
        border: 0;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
        background: #513287;
        color: white;
    }
    [data-testid="stForm"] {
        background: #1a2232;
        border: 1px solid #343d52;
        border-radius: 16px;
        padding: 1.15rem;
    }
    [data-testid="stExpander"] {
        background: #1a2232;
        border-color: #343d52;
        border-radius: 12px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #1a2232;
        border-color: #343d52 !important;
    }
    [data-testid="stAlert"] p { color: inherit !important; }
    [data-baseweb="tab"] { color: #cdd5e7; }
    [aria-selected="true"][data-baseweb="tab"] { color: #b99af2; }
    [data-baseweb="input"] { background: #101622; }
    [data-baseweb="input"] input { color: #ffffff; }
    [data-baseweb="input"] input::placeholder { color: #aeb7ca; opacity: 1; }
    .section-label {
        color: #a98be5;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 1.6rem 0 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ✦ Local PDF AI")
    st.caption("PRIVATE DOCUMENT Q&A")
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("① Upload PDF")
    st.markdown("② Extract & chunk")
    st.markdown("③ Hybrid search")
    st.markdown("④ Generate answer")
    st.markdown("⑤ Verify sources")
    st.divider()
    st.caption("ACTIVE MODEL")
    st.code(OLLAMA_MODEL, language=None)
    st.caption("Your document and questions stay on this computer.")

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Version 1 · Local RAG</div>
        <h1>Ask your PDF. Verify every answer.</h1>
        <p>Search a document with local embeddings and get concise answers backed by page-level evidence.</p>
        <span class="privacy-pill">● 100% local processing</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">01 · Document</div>', unsafe_allow_html=True)
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload a text-based PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="The PDF is processed in memory and is not intentionally saved.",
    )

if uploaded_file is None:
    st.info("Choose a PDF above to prepare it for questions.", icon="↗️")
    starter_1, starter_2, starter_3 = st.columns(3)
    with starter_1:
        with st.container(border=True):
            st.markdown("#### Private by design")
            st.caption("PDF processing, search, and answer generation run locally.")
    with starter_2:
        with st.container(border=True):
            st.markdown("#### Source-backed")
            st.caption("Every generated answer points to the evidence used.")
    with starter_3:
        with st.container(border=True):
            st.markdown("#### Easy to inspect")
            st.caption("Open retrieved chunks and original page text at any time.")
    st.stop()

try:
    pages = extract_pdf_pages(uploaded_file.getvalue(), uploaded_file.name)
except Exception as error:
    st.error(f"I could not read this PDF: {error}")
    st.stop()

if not pages:
    st.warning(
        "No selectable text was found. This may be a scanned PDF; "
        "image-based PDFs will need OCR in a later step."
    )
    st.stop()

word_count = sum(len(page.text.split()) for page in pages)
chunks = chunk_pdf_pages(pages)

with st.spinner("Creating local embeddings..."):
    try:
        embedded_chunks = embed_chunks(chunks, get_embedding_model())
    except Exception as error:
        st.error(f"I could not create embeddings: {error}")
        st.stop()

embedding_dimensions = (
    len(embedded_chunks[0].embedding) if embedded_chunks else 0
)

st.markdown('<div class="section-label">02 · Document ready</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pages with text", len(pages))
col2.metric("Approximate words", f"{word_count:,}")
col3.metric("Searchable chunks", len(chunks))
col4.metric("Numbers per embedding", embedding_dimensions)

st.success("Document indexed successfully and ready for questions.", icon="✅")

with st.expander("Behind the scenes: chunks and embeddings"):
    st.write(
        "Each chunk contains up to 150 words, with a 30-word overlap between "
        "neighbours so important context is less likely to be split."
    )
    st.write(
        "Each chunk is represented by 384 numbers. Here are the first "
        "eight numbers for the first chunk:"
    )
    st.code(
        str([round(value, 4) for value in embedded_chunks[0].embedding[:8]])
    )

st.markdown('<div class="section-label">03 · Ask your document</div>', unsafe_allow_html=True)
st.subheader("What would you like to know?")

with st.form("semantic_search_form"):
    question = st.text_input(
        "Ask a question about this document",
        placeholder="For example: What is the annual leave policy?",
    )
    search_submitted = st.form_submit_button("Ask with Qwen3 8B  →")

if search_submitted:
    try:
        results = semantic_search(question, embedded_chunks, get_embedding_model())
    except ValueError as error:
        st.warning(str(error))
    else:
        with st.spinner(
            f"{OLLAMA_MODEL} is writing an evidence-grounded answer..."
        ):
            try:
                answer = generate_grounded_answer(question, results)
            except (OllamaError, ValueError) as error:
                st.error(str(error))
            else:
                st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"### {answer}")

        st.markdown('<div class="section-label">Evidence</div>', unsafe_allow_html=True)
        st.subheader("Sources used for this answer")
        st.caption(
            "These are the three passages retrieved before answer generation."
        )

        for rank, result in enumerate(results, start=1):
            chunk = result.embedded_chunk.chunk
            with st.container(border=True):
                st.markdown(
                    f"**Source {rank}: {chunk.document_name} — page "
                    f"{chunk.page_number}, chunk {chunk.chunk_number}**"
                )
                st.caption(f"Semantic similarity · {result.score:.3f}")
                st.write(chunk.text)

st.markdown('<div class="section-label">04 · Inspect</div>', unsafe_allow_html=True)
st.subheader("Document details")

chunk_tab, page_tab = st.tabs(["Searchable chunks", "Original page text"])

with chunk_tab:
    for chunk in chunks:
        with st.expander(
            f"Page {chunk.page_number} · chunk {chunk.chunk_number}"
        ):
            st.write(chunk.text)

with page_tab:
    for page in pages:
        with st.expander(f"Page {page.page_number} · {page.document_name}"):
            st.text(page.text)
