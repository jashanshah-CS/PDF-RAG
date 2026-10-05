"""Streamlit interface for local PDF retrieval and answer generation."""

import streamlit as st

from src.rag_project.chunker import chunk_documents
from src.rag_project.csv_loader import (
    CsvLoadError,
    answer_csv_question,
    load_csv_file,
    looks_like_csv_calculation,
)
from src.rag_project.embeddings import embed_chunks, load_embedding_model
from src.rag_project.generator import (
    OLLAMA_MODEL,
    OllamaError,
    generate_grounded_answer,
)
from src.rag_project.pdf_loader import extract_pdf_files
from src.rag_project.search import semantic_search
from src.rag_project.website_loader import WebsiteLoadError, crawl_website


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load the model once and reuse it across Streamlit reruns."""
    return load_embedding_model()


@st.cache_data(show_spinner=False, ttl=3600)
def get_website_documents(url: str):
    """Load a bounded approved website once and reuse it across reruns."""
    return crawl_website(url, max_pages=5)


st.set_page_config(
    page_title="Local Source AI",
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
    st.markdown("## ✦ Local Source AI")
    st.caption("PRIVATE DOCUMENT Q&A")
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("① Add PDFs, website, or CSV")
    st.markdown("② Extract & chunk")
    st.markdown("③ Hybrid search")
    st.markdown("④ Generate answer")
    st.markdown("⑤ Verify sources")
    st.divider()
    st.caption("ACTIVE MODEL")
    st.code(OLLAMA_MODEL, language=None)
    st.caption("Your documents and questions stay on this computer.")

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Version 2 · Unified sources</div>
        <h1>Ask your sources. Verify every answer.</h1>
        <p>Search PDFs, approved webpages, and CSV data together, with precise source evidence.</p>
        <span class="privacy-pill">● 100% local processing</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">01 · Sources</div>', unsafe_allow_html=True)
with st.container(border=True):
    pdf_tab, website_tab, csv_tab = st.tabs(["PDF documents", "Website", "CSV data"])
    with pdf_tab:
        uploaded_files = st.file_uploader(
            "Upload one or more text-based PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="PDFs are processed in memory and are not intentionally saved.",
        )
    with website_tab:
        with st.form("website_source_form"):
            entered_website_url = st.text_input(
                "Public webpage URL",
                value=st.session_state.get("website_source_url", ""),
                placeholder="https://example.com/benefits",
                help="Loads this page and up to four linked pages on the same hostname.",
            )
            website_submitted = st.form_submit_button("Add website pages")

        if website_submitted:
            st.session_state["website_source_url"] = entered_website_url.strip()

        website_url = st.session_state.get("website_source_url", "")
        if website_url:
            st.caption(f"Added website: {website_url}")
            if st.button("Remove website", use_container_width=True):
                st.session_state.pop("website_source_url", None)
                st.rerun()
    with csv_tab:
        uploaded_csv = st.file_uploader(
            "Upload one UTF-8 CSV file",
            type=["csv"],
            accept_multiple_files=False,
            help="CSV files are validated and limited to 2 MB and 10,000 rows.",
            key="csv_source_uploader",
        )

website_url = st.session_state.get("website_source_url", "")
if not uploaded_files and not website_url and uploaded_csv is None:
    st.info("Upload PDFs or CSV data, or add a public webpage to begin.", icon="↗️")
    starter_1, starter_2, starter_3 = st.columns(3)
    with starter_1:
        with st.container(border=True):
            st.markdown("#### Private by design")
            st.caption("PDF processing and AI stay local; webpages are downloaded from their URL.")
    with starter_2:
        with st.container(border=True):
            st.markdown("#### Source-backed")
            st.caption("Every generated answer points to the evidence used.")
    with starter_3:
        with st.container(border=True):
            st.markdown("#### Easy to inspect")
            st.caption("Open retrieved chunks and extracted source text at any time.")
    st.stop()

documents = []
csv_tables = []
try:
    if uploaded_files:
        documents.extend(
            extract_pdf_files(
                (uploaded_file.name, uploaded_file.getvalue())
                for uploaded_file in uploaded_files
            )
        )
except Exception as error:
    st.error(f"I could not read the uploaded PDFs: {error}")
    st.stop()

if uploaded_csv is not None:
    try:
        csv_table = load_csv_file(uploaded_csv.getvalue(), uploaded_csv.name)
    except CsvLoadError as error:
        st.error(f"I could not read the uploaded CSV: {error}")
        st.stop()
    csv_tables.append(csv_table)
    documents.extend(csv_table.to_documents())

if website_url:
    with st.spinner("Discovering and extracting up to five approved website pages..."):
        try:
            documents.extend(get_website_documents(website_url))
        except WebsiteLoadError as error:
            st.error(f"I could not add that website: {error}")
            st.stop()

if not documents:
    st.warning(
        "No readable text was found. Uploaded files may be scanned PDFs; "
        "image-based PDFs will need OCR in a later step."
    )
    st.stop()

word_count = sum(len(document.text.split()) for document in documents)
chunks = chunk_documents(documents)

with st.spinner("Creating local embeddings..."):
    try:
        embedded_chunks = embed_chunks(chunks, get_embedding_model())
    except Exception as error:
        st.error(f"I could not create embeddings: {error}")
        st.stop()

source_count = len({document.document_id for document in documents})
pdf_page_count = sum(
    document.location.page_number is not None for document in documents
)
website_pages = {
    document.location.url
    for document in documents
    if document.location.url is not None
}
csv_row_count = sum(len(table.rows) for table in csv_tables)

st.markdown('<div class="section-label">02 · Documents ready</div>', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Sources", source_count)
col2.metric("PDF pages", pdf_page_count)
col3.metric("CSV rows", csv_row_count)
col4.metric("Approximate words", f"{word_count:,}")
col5.metric("Searchable chunks", len(chunks))

label = "source" if source_count == 1 else "sources"
st.success(
    f"{source_count} {label} indexed successfully and ready for questions.",
    icon="✅",
)
if website_pages:
    with st.expander(f"Website pages indexed ({len(website_pages)})"):
        for page_url in sorted(website_pages):
            st.markdown(f"- [{page_url}]({page_url})")
if csv_tables:
    with st.expander(f"CSV preview · {csv_tables[0].source_name}"):
        st.dataframe(csv_tables[0].preview(), use_container_width=True)

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

st.markdown('<div class="section-label">03 · Ask your sources</div>', unsafe_allow_html=True)
st.subheader("What would you like to know?")

with st.form("semantic_search_form"):
    question = st.text_input(
        "Ask a question about these sources",
        placeholder="For example: What is the annual leave policy?",
    )
    search_submitted = st.form_submit_button("Ask your sources  →")

if search_submitted:
    structured_answer = answer_csv_question(question, csv_tables)
    if structured_answer is not None:
        st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"### {structured_answer.answer}")
            st.caption("Calculated directly from the CSV; no language model arithmetic used.")

        st.markdown('<div class="section-label">Evidence</div>', unsafe_allow_html=True)
        st.subheader("CSV rows used for this answer")
        visible_citations = structured_answer.citations[:25]
        for citation in visible_citations:
            with st.container(border=True):
                st.markdown(f"**{citation.label()}**")
                st.write(citation.values)
        hidden_count = len(structured_answer.citations) - len(visible_citations)
        if hidden_count:
            st.caption(f"{hidden_count:,} additional contributing rows are not displayed.")
    elif csv_tables and looks_like_csv_calculation(question):
        st.warning(
            "I recognized this as a CSV calculation but could not map it safely. "
            "Use the exact column names shown in the CSV preview."
        )
    else:
        try:
            results = semantic_search(question, embedded_chunks, get_embedding_model())
        except ValueError as error:
            st.warning(str(error))
            results = []

    if structured_answer is None and not (
        csv_tables and looks_like_csv_calculation(question)
    ) and results:
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
                    f"**Source {rank}: {chunk.citation_label()}, "
                    f"chunk {chunk.chunk_number}**"
                )
                st.caption(f"Semantic similarity · {result.score:.3f}")
                st.write(chunk.text)
                if chunk.location.url:
                    st.link_button("Open webpage", chunk.location.url)

st.markdown('<div class="section-label">04 · Inspect</div>', unsafe_allow_html=True)
st.subheader("Document details")

chunk_tab, page_tab = st.tabs(["Searchable chunks", "Extracted source text"])

with chunk_tab:
    for chunk in chunks:
        with st.expander(
            f"{chunk.location.label().title()} · chunk {chunk.chunk_number}"
        ):
            st.write(chunk.text)

with page_tab:
    for document in documents:
        with st.expander(
            f"{document.location.label()} · {document.source_name}"
        ):
            st.text(document.text)
            if document.location.url:
                st.link_button("Open original webpage", document.location.url)
