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


st.set_page_config(page_title="Local PDF Q&A", page_icon="📄")

st.title("Local PDF Q&A")
st.caption("Version 1 · Step 5: answer questions with local PDF evidence")

uploaded_file = st.file_uploader(
    "Choose one PDF",
    type=["pdf"],
    accept_multiple_files=False,
    help="The file is processed locally by this app.",
)

if uploaded_file is None:
    st.info("Upload a text-based PDF to begin.")
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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Pages with text", len(pages))
col2.metric("Approximate words", f"{word_count:,}")
col3.metric("Searchable chunks", len(chunks))
col4.metric("Numbers per embedding", embedding_dimensions)

st.success("Text extracted, chunked, and embedded locally.")
st.info(
    "Each chunk contains up to 150 words. Neighbouring chunks share 30 words "
    "so important context is less likely to be cut in half."
)

with st.expander("What does one embedding look like?"):
    st.write(
        "Each chunk is represented by 384 numbers. Here are the first "
        "eight numbers for the first chunk:"
    )
    st.code(
        str([round(value, 4) for value in embedded_chunks[0].embedding[:8]])
    )

st.subheader("Ask the PDF")

with st.form("semantic_search_form"):
    question = st.text_input(
        "Ask a question about this document",
        placeholder="For example: What programming languages are mentioned?",
    )
    search_submitted = st.form_submit_button("Generate answer")

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
                st.subheader("Answer")
                st.markdown(answer)

        st.subheader("Evidence used")
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
                st.caption(f"Cosine similarity: {result.score:.3f}")
                st.write(chunk.text)

st.subheader("Chunk preview")

for chunk in chunks:
    with st.expander(
        f"{chunk.document_name} · page {chunk.page_number} · "
        f"chunk {chunk.chunk_number}"
    ):
        st.write(chunk.text)

st.subheader("Original page text")

for page in pages:
    with st.expander(f"{page.document_name} · page {page.page_number}"):
        st.text(page.text)
