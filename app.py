"""Streamlit interface for persistent local document retrieval and answers."""

from dataclasses import replace

import streamlit as st

from src.rag_project.chunker import chunk_documents
from src.rag_project.embeddings import embed_chunks, load_embedding_model
from src.rag_project.generator import (
    OLLAMA_MODEL,
    OllamaError,
    generate_grounded_answer,
)
from src.rag_project.pdf_loader import extract_pdf_files
from src.rag_project.search import semantic_search
from src.rag_project.website_loader import WebsiteLoadError, crawl_website
from src.rag_project.vector_store import (
    PersistentVectorStore,
    filter_chunks_by_source_keys,
    pdf_source_key,
    website_source_key,
)
from src.rag_project.history_store import QuestionHistoryStore


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load the model once and reuse it across Streamlit reruns."""
    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def get_vector_store():
    """Open the persistent local ChromaDB collection once per app process."""
    return PersistentVectorStore()


@st.cache_resource(show_spinner=False)
def get_history_store():
    """Open the readable local SQLite question history."""
    return QuestionHistoryStore()


def render_history(section_number: str) -> None:
    """Render searchable, exportable question-and-answer history."""
    history_store = get_history_store()
    st.markdown(
        f'<div class="section-label">{section_number} · History</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Previous questions and answers")
    search = st.text_input(
        "Search history",
        placeholder="Search questions or answers",
        key=f"history-search-{section_number}",
    )
    entries = history_store.list(search)
    history_col, export_col = st.columns([3, 1])
    history_col.caption(
        f"{len(entries)} saved {'entry' if len(entries) == 1 else 'entries'} shown"
    )
    export_col.download_button(
        "Export history",
        data=history_store.export_json(),
        file_name="rag-question-history.json",
        mime="application/json",
        use_container_width=True,
    )

    if not entries:
        st.info("No matching question history yet.")
    for entry in entries:
        local_time = entry.created_at.astimezone().strftime("%d %b %Y %H:%M")
        with st.expander(f"{local_time} · {entry.question}"):
            st.markdown(entry.answer)

    with st.expander("History controls"):
        confirmed = st.checkbox(
            "I understand this permanently deletes all saved questions and answers.",
            key=f"clear-history-confirm-{section_number}",
        )
        if st.button(
            "Clear question history",
            disabled=not confirmed,
            key=f"clear-history-{section_number}",
            use_container_width=True,
        ):
            history_store.clear()
            st.rerun()


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
    st.markdown("Add your documents, choose what to search, and ask a question.")
    st.caption("Your documents and questions stay on this computer.")

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Version 2 · Unified sources</div>
        <h1>Ask your documents.</h1>
        <p>Search your PDFs and approved webpages together and get a clear answer.</p>
        <span class="privacy-pill">● 100% local processing</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">01 · Sources</div>', unsafe_allow_html=True)
with st.container(border=True):
    pdf_tab, website_tab = st.tabs(["PDF documents", "Website"])
    with pdf_tab:
        uploaded_files = st.file_uploader(
            "Upload one or more text-based PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help=(
                "Your PDF is saved locally. Uploading a changed PDF with the "
                "same filename replaces its previous version."
            ),
        )
        index_pdfs = st.button(
            "Add or update PDFs",
            disabled=not uploaded_files,
            use_container_width=True,
        )
    with website_tab:
        with st.form("website_source_form"):
            entered_website_url = st.text_input(
                "Public webpage URL",
                placeholder="https://example.com/benefits",
                help="Loads this page and up to four linked pages on the same hostname.",
            )
            website_submitted = st.form_submit_button("Add or refresh website")

store = get_vector_store()

if index_pdfs and uploaded_files:
    with st.spinner("Processing and saving PDFs locally..."):
        try:
            for uploaded_file in uploaded_files:
                pdf_documents = extract_pdf_files(
                    [(uploaded_file.name, uploaded_file.getvalue())]
                )
                if not pdf_documents:
                    st.warning(f"No readable text was found in {uploaded_file.name}.")
                    continue
                pdf_chunks = chunk_documents(pdf_documents)
                embedded_pdf_chunks = embed_chunks(
                    pdf_chunks, get_embedding_model()
                )
                store.replace_source(
                    pdf_source_key(uploaded_file.name), embedded_pdf_chunks
                )
            st.success("PDF index saved. It will remain available after restart.")
        except Exception as error:
            st.error(f"I could not index the uploaded PDFs: {error}")

if website_submitted:
    website_url = entered_website_url.strip()
    if not website_url:
        st.warning("Enter a public website URL first.")
    else:
        with st.spinner("Processing and saving website pages..."):
            try:
                website_documents = [
                    replace(
                        document,
                        metadata={**document.metadata, "root_url": website_url},
                    )
                    for document in crawl_website(website_url, max_pages=5)
                ]
                website_chunks = chunk_documents(website_documents)
                embedded_website_chunks = embed_chunks(
                    website_chunks, get_embedding_model()
                )
                store.replace_source(
                    website_source_key(website_url), embedded_website_chunks
                )
                st.success("Website index saved. Use the same URL to refresh it.")
            except WebsiteLoadError as error:
                st.error(f"I could not add that website: {error}")
            except Exception as error:
                st.error(f"I could not index that website: {error}")

stored_sources = store.list_sources()

embedded_chunks = store.load_all()
if not embedded_chunks:
    st.info("Upload PDFs or add a public webpage to begin.", icon="↗️")
    starter_1, starter_2, starter_3 = st.columns(3)
    with starter_1:
        with st.container(border=True):
            st.markdown("#### Private by design")
            st.caption("PDF processing and AI stay local; webpages are downloaded from their URL.")
    with starter_2:
        with st.container(border=True):
            st.markdown("#### Ask naturally")
            st.caption("Ask in plain language and receive a clear answer.")
    with starter_3:
        with st.container(border=True):
            st.markdown("#### Saved locally")
            st.caption("Your documents return automatically after an app restart.")
    render_history("02")
    st.stop()

st.success("Your documents are ready for questions.", icon="✅")

st.markdown('<div class="section-label">02 · Ask a question</div>', unsafe_allow_html=True)
st.subheader("What would you like to know?")

available_source_keys = [source.source_key for source in stored_sources]
source_labels = {
    source.source_key: (
        f"{'Website' if source.root_url else 'PDF'} · {source.source_name}"
    )
    for source in stored_sources
}
selection_key = "selected_source_keys"
if selection_key not in st.session_state:
    st.session_state[selection_key] = available_source_keys
else:
    st.session_state[selection_key] = [
        key
        for key in st.session_state[selection_key]
        if key in available_source_keys
    ]

select_all_col, clear_selection_col = st.columns(2)
if select_all_col.button("Select all sources", use_container_width=True):
    st.session_state[selection_key] = available_source_keys
    st.rerun()
if clear_selection_col.button("Clear source selection", use_container_width=True):
    st.session_state[selection_key] = []
    st.rerun()

selected_source_keys = st.multiselect(
    "Sources to search",
    options=available_source_keys,
    format_func=lambda key: source_labels[key],
    key=selection_key,
    help="Only selected sources can contribute retrieved evidence or answers.",
)
selected_source_names = tuple(
    source_labels[key] for key in selected_source_keys
)
if not selected_source_names:
    st.warning("Select at least one source before asking a question.")

searchable_chunks = filter_chunks_by_source_keys(
    embedded_chunks, set(selected_source_keys)
)

with st.form("semantic_search_form"):
    question = st.text_input(
        "Ask a question about these sources",
        placeholder="For example: What is the annual leave policy?",
    )
    search_submitted = st.form_submit_button(
        "Ask with Qwen3 8B  →", disabled=not searchable_chunks
    )

if search_submitted:
    try:
        results = semantic_search(question, searchable_chunks, get_embedding_model())
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
                try:
                    get_history_store().add(
                        question,
                        answer,
                        results,
                        selected_sources=selected_source_names,
                    )
                except Exception as error:
                    st.warning(f"The answer was created but history could not be saved: {error}")
                st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"### {answer}")

render_history("03")
