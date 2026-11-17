"""Streamlit interface for persistent local document retrieval and answers."""

from dataclasses import replace

import streamlit as st

from src.rag_project.chunker import chunk_documents
from src.rag_project.conversation_store import (
    Conversation,
    ConversationStore,
    MessageCitation,
)
from src.rag_project.conversation_ui import (
    choose_active_conversation,
    conversation_title,
    message_citations,
    reconcile_conversation_sources,
)
from src.rag_project.embeddings import embed_chunks, load_embedding_model
from src.rag_project.generator import (
    OLLAMA_MODEL,
    OllamaError,
    generate_grounded_answer,
)
from src.rag_project.pdf_loader import extract_pdf_files
from src.rag_project.reranker import RerankedRetriever
from src.rag_project.search import SearchResult
from src.rag_project.website_loader import WebsiteLoadError, crawl_website
from src.rag_project.vector_store import (
    PersistentVectorStore,
    filter_chunks_by_source_keys,
    pdf_source_key,
    website_source_key,
)
from src.rag_project.history_store import HistoryCitation, QuestionHistoryStore
from src.rag_project.citations import cited_source_numbers
from src.rag_project.startup import run_startup_checks


SOURCE_SELECTION_KEY = "selected_source_keys"
SOURCE_SELECTION_UI_VERSION = 3
ACTIVE_CONVERSATION_KEY = "active_conversation_id"
CONVERSATION_SOURCE_OWNER_KEY = "conversation_source_owner"
CONVERSATION_SELECTOR_VERSION_KEY = "conversation_selector_version"


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


@st.cache_resource(show_spinner=False)
def get_conversation_store():
    """Open the Version 4 multi-turn conversation database."""
    return ConversationStore()


@st.cache_data(ttl=30, show_spinner=False)
def get_startup_checks():
    """Recheck local services periodically and present actionable status."""
    return run_startup_checks()


def render_answer_sources(answer: str, results: list[SearchResult]) -> None:
    """Render the evidence actually cited by an answer as expandable cards."""
    source_numbers = cited_source_numbers(answer, len(results))
    if not source_numbers:
        return
    st.markdown(
        '<div class="citation-intro"><strong>Sources used</strong>'
        '<span>Select a source to view the supporting passage.</span></div>',
        unsafe_allow_html=True,
    )
    for number in source_numbers:
        chunk = results[number - 1].embedded_chunk.chunk
        source_type = "Webpage" if chunk.location.url else "PDF"
        with st.expander(f"Source {number} · {chunk.source_name}"):
            st.caption(f"{source_type} · {chunk.location.label()}")
            st.write(chunk.text)
            if chunk.location.url:
                st.link_button(
                    "Open original webpage ↗",
                    chunk.location.url,
                    use_container_width=True,
                )


def render_saved_sources(citations: tuple[HistoryCitation, ...]) -> None:
    """Render saved citation evidence inside a history entry."""
    if not citations:
        return
    st.markdown("**Sources used**")
    for position, citation in enumerate(citations, start=1):
        number = citation.source_number or position
        with st.expander(f"Source {number} · {citation.source_name}"):
            st.caption(f"{citation.source_type.title()} · {citation.location}")
            if citation.excerpt:
                st.write(citation.excerpt)
            if citation.url:
                st.link_button(
                    "Open original webpage ↗",
                    citation.url,
                    use_container_width=True,
                )


def render_conversation_sources(citations: tuple[MessageCitation, ...]) -> None:
    """Render evidence persisted with one assistant chat message."""
    if not citations:
        return
    st.markdown("**Sources used**")
    for citation in citations:
        with st.expander(f"Source {citation.source_number} · {citation.source_name}"):
            st.caption(f"{citation.source_type.title()} · {citation.location}")
            st.write(citation.excerpt)
            if citation.url:
                st.link_button(
                    "Open original webpage ↗",
                    citation.url,
                    use_container_width=True,
                )


def render_conversation_timeline(
    conversation_store: ConversationStore,
    conversation: Conversation,
) -> None:
    """Render one persisted conversation in chronological order."""
    st.markdown(
        '<div class="section-label">03 · Conversation</div>',
        unsafe_allow_html=True,
    )
    st.subheader(conversation.title)
    messages = conversation_store.list_messages(conversation.conversation_id)
    if not messages:
        st.info("This conversation is empty. Ask a question to begin.")
        return
    status_labels = {
        "pending": "Generating…",
        "failed": "Generation failed",
        "cancelled": "Generation cancelled",
    }
    for message in messages:
        with st.chat_message(message.role):
            if message.status in status_labels:
                st.caption(status_labels[message.status])
            st.markdown(message.content)
            if message.role == "assistant":
                render_conversation_sources(message.citations)


def render_conversation_sidebar(
    conversation_store: ConversationStore,
) -> Conversation:
    """Render conversation selection and management controls in the sidebar."""
    st.markdown("### Conversations")
    if st.button("＋ New conversation", use_container_width=True):
        created = conversation_store.create()
        st.session_state[ACTIVE_CONVERSATION_KEY] = created.conversation_id
        st.session_state[CONVERSATION_SELECTOR_VERSION_KEY] = (
            st.session_state.get(CONVERSATION_SELECTOR_VERSION_KEY, 0) + 1
        )
        st.session_state[CONVERSATION_SOURCE_OWNER_KEY] = None
        st.rerun()

    search = st.text_input(
        "Find a conversation",
        placeholder="Search titles or messages",
        key="conversation-search",
    )
    all_conversations = conversation_store.list()
    if not all_conversations:
        all_conversations = [conversation_store.create()]
    active_id = choose_active_conversation(
        all_conversations,
        st.session_state.get(ACTIVE_CONVERSATION_KEY),
    )
    assert active_id is not None
    conversations = (
        conversation_store.list(search) if search.strip() else all_conversations
    )
    active_item = next(
        item for item in all_conversations if item.conversation_id == active_id
    )
    if active_id not in {item.conversation_id for item in conversations}:
        conversations.insert(0, active_item)
        if search.strip():
            st.caption("No other matching conversations.")
    options = [
        (item.conversation_id, item.title)
        for item in conversations
    ]
    active_option = next(option for option in options if option[0] == active_id)
    selected_option = st.selectbox(
        "Current conversation",
        options=options,
        index=options.index(active_option),
        format_func=lambda option: option[1],
        key=(
            "conversation-selector-"
            f"{st.session_state.get(CONVERSATION_SELECTOR_VERSION_KEY, 0)}"
        ),
    )
    selected_id = selected_option[0]
    if selected_id != st.session_state.get(ACTIVE_CONVERSATION_KEY):
        st.session_state[ACTIVE_CONVERSATION_KEY] = selected_id
        st.session_state[CONVERSATION_SOURCE_OWNER_KEY] = None
        st.rerun()

    active = conversation_store.get(selected_id)
    assert active is not None
    with st.expander("Manage current conversation"):
        with st.form(f"rename-conversation-{selected_id}"):
            title = st.text_input("Conversation title", value=active.title)
            if st.form_submit_button("Rename", use_container_width=True):
                conversation_store.rename(selected_id, title)
                st.rerun()
        confirmed = st.checkbox(
            "Permanently delete this conversation",
            key=f"delete-conversation-confirm-{selected_id}",
        )
        if st.button(
            "Delete conversation",
            disabled=not confirmed,
            key=f"delete-conversation-{selected_id}",
            use_container_width=True,
        ):
            conversation_store.delete(selected_id)
            remaining = conversation_store.list(limit=1)
            replacement = remaining[0] if remaining else conversation_store.create()
            st.session_state[ACTIVE_CONVERSATION_KEY] = replacement.conversation_id
            st.session_state[CONVERSATION_SELECTOR_VERSION_KEY] = (
                st.session_state.get(CONVERSATION_SELECTOR_VERSION_KEY, 0) + 1
            )
            st.session_state[CONVERSATION_SOURCE_OWNER_KEY] = None
            st.rerun()
    return active


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
            render_saved_sources(entry.citations)

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

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

dark_mode = bool(st.session_state["dark_mode"])

st.markdown(
    """
    <style>
    :root {
        --ink: #17201d;
        --muted: #68716d;
        --paper: #f4f1e9;
        --surface: #fffdf8;
        --line: #d7d3c9;
        --forest: #18372f;
        --forest-soft: #285448;
        --clay: #c55e3f;
        --clay-dark: #9f442c;
        --sage: #dce5df;
        --sidebar: #18372f;
        --sidebar-panel: #21473c;
        --sidebar-line: #31594e;
        --header: rgba(244, 241, 233, 0.92);
        --button-ink: #ffffff;
        --uploader: #f7f4ed;
        --upload-border: #aaa79e;
        --disabled: #e9e5db;
        --disabled-line: #d2cdc1;
        --disabled-ink: #606864;
        --placeholder: #66706b;
        --alert-bg: #edf3ee;
    }
    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: "Aptos", "Segoe UI", sans-serif;
    }
    [data-testid="stHeader"] { background: var(--header); }
    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #f4f1e9; }
    [data-testid="stSidebar"] .stCaptionContainer { color: #bfcfc8; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: var(--sidebar-panel) !important;
        border-color: var(--sidebar-line) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
        color: #f4f1e9 !important;
    }
    .block-container {
        max-width: 1080px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4 { color: var(--ink); }
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        color: var(--muted);
    }
    .workspace-head {
        padding: 0.3rem 0 1.8rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.6rem;
    }
    .workspace-kicker {
        color: var(--clay-dark);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .workspace-title-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1.5rem;
    }
    .workspace-head h1 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.2rem, 5vw, 3.7rem);
        line-height: 1.02;
        letter-spacing: -0.045em;
        font-weight: 500;
        margin: 0;
        color: var(--ink);
    }
    .workspace-head p {
        color: var(--muted);
        max-width: 640px;
        font-size: 1.02rem;
        line-height: 1.65;
        margin: 0.8rem 0 0;
    }
    .local-mark {
        color: var(--forest);
        border-bottom: 2px solid var(--clay);
        padding-bottom: 0.2rem;
        white-space: nowrap;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    [data-testid="stFileUploader"] {
        background: var(--uploader);
        border: 1px dashed var(--upload-border);
        border-radius: 6px;
        color: var(--ink);
    }
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] button {
        color: var(--ink) !important;
    }
    [data-testid="stFileUploader"] ul,
    [data-testid="stFileUploader"] li {
        background: var(--surface) !important;
        color: var(--ink) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface) !important;
        border-color: var(--upload-border) !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--paper);
        border: 1px solid var(--line);
        color: var(--forest) !important;
    }
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button,
    [data-testid="stDownloadButton"] > button {
        border-radius: 5px;
        min-height: 2.8rem;
        font-weight: 700;
        background: var(--forest);
        color: var(--button-ink);
        border: 1px solid var(--forest);
        box-shadow: none;
    }
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        background: var(--forest-soft);
        border-color: var(--forest-soft);
        color: var(--button-ink);
    }
    .stButton > button *,
    [data-testid="stFormSubmitButton"] > button *,
    [data-testid="stDownloadButton"] > button * {
        color: inherit !important;
    }
    .stButton > button:focus { box-shadow: 0 0 0 3px rgba(197,94,63,0.18); }
    .stButton > button:disabled,
    [data-testid="stFormSubmitButton"] > button:disabled {
        background: var(--disabled) !important;
        border-color: var(--disabled-line) !important;
        color: var(--disabled-ink) !important;
        opacity: 1 !important;
    }
    [data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.25rem;
    }
    [data-testid="stExpander"] {
        background: var(--surface);
        border-color: var(--line);
        border-radius: 6px;
    }
    [data-testid="stMain"] [data-testid="stExpander"] > details > summary {
        background: var(--surface) !important;
        color: var(--ink) !important;
    }
    [data-testid="stMain"] [data-testid="stExpander"] > details > summary * {
        color: var(--ink) !important;
    }
    [data-testid="stMain"] [data-testid="stExpander"] > details > summary:hover {
        background: var(--sage) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-color: var(--line) !important;
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] {
        background: var(--alert-bg) !important;
        border: 1px solid var(--line) !important;
        border-radius: 5px;
    }
    [data-testid="stAlert"] * { color: var(--ink) !important; }
    [data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color: var(--ink);
    }
    [data-baseweb="tab"] { color: var(--muted); }
    [aria-selected="true"][data-baseweb="tab"] { color: var(--forest); }
    [data-baseweb="input"], [data-baseweb="select"] {
        background: var(--surface);
        border-color: var(--line);
    }
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea { color: var(--ink) !important; }
    [data-baseweb="input"] input::placeholder,
    [data-baseweb="textarea"] textarea::placeholder {
        color: var(--placeholder) !important;
        opacity: 1;
    }
    [data-baseweb="tag"] { background-color: var(--sage) !important; }
    [data-baseweb="tag"] span { color: var(--forest) !important; }
    [data-testid="stPopover"] button {
        background: var(--surface);
        color: var(--forest);
        border: 1px solid var(--forest);
    }
    [data-testid="stPopover"] button:hover {
        background: var(--sage);
        color: var(--forest);
    }
    .section-label {
        color: var(--clay-dark);
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin: 2.1rem 0 0.4rem;
    }
    .source-summary {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: var(--muted);
        font-size: 0.88rem;
        margin: 0.15rem 0 0.75rem;
    }
    .source-summary strong { color: var(--ink); }
    .source-dot {
        width: 8px;
        height: 8px;
        background: var(--clay);
        border-radius: 50%;
        display: inline-block;
    }
    .citation-intro {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin: 1.15rem 0 0.55rem;
        padding-top: 1rem;
        border-top: 1px solid var(--line);
        color: var(--ink);
    }
    .citation-intro strong {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .citation-intro span {
        color: var(--muted);
        font-size: 0.82rem;
    }
    h2, h3 { letter-spacing: -0.02em; }
    @media (max-width: 760px) {
        .block-container { padding: 1.4rem 1rem 3rem; }
        .workspace-title-row { display: block; }
        .local-mark { display: inline-block; margin-top: 1rem; }
        .workspace-head h1 { font-size: 2.45rem; }
        .citation-intro { display: block; }
        .citation-intro span { display: block; margin-top: 0.25rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if dark_mode:
    st.markdown(
        """
        <style>
        :root {
            --ink: #ecebe4;
            --muted: #aab5af;
            --paper: #101613;
            --surface: #18211d;
            --line: #34423c;
            --forest: #91bca8;
            --forest-soft: #afd0c0;
            --clay: #df7958;
            --clay-dark: #ee9678;
            --sage: #273a32;
            --sidebar: #0b1c16;
            --sidebar-panel: #163127;
            --sidebar-line: #2b5043;
            --header: rgba(16, 22, 19, 0.92);
            --button-ink: #102018;
            --uploader: #141c18;
            --upload-border: #536059;
            --disabled: #222b27;
            --disabled-line: #39443f;
            --disabled-ink: #aab4af;
            --placeholder: #aab4af;
            --alert-bg: #1c2923;
        }
        [data-testid="stToolbar"] { color: var(--ink); }
        [data-testid="stFileUploaderDropzone"] button {
            background: #202b26;
            color: var(--ink) !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"] {
            background: var(--surface) !important;
            color: var(--ink) !important;
        }
        [role="option"] { color: var(--ink) !important; }
        [role="option"]:hover { background: var(--sage) !important; }
        [data-baseweb="textarea"] textarea,
        [data-baseweb="input"] input {
            background: var(--surface) !important;
            color: var(--ink) !important;
        }
        [data-testid="stMarkdownContainer"] code {
            background: #222d28;
            color: #f1c2b2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

conversation_store: ConversationStore | None = None
active_conversation: Conversation | None = None

with st.sidebar:
    st.markdown("## ✦ Local PDF AI")
    st.caption("PRIVATE DOCUMENT Q&A")
    st.toggle(
        "Dark mode",
        key="dark_mode",
        help="Switch between the light editorial theme and a low-glare dark theme.",
    )
    st.divider()
    st.markdown("Add your documents, choose what to search, and ask a question.")
    st.caption("Your documents and questions stay on this computer.")
    with st.expander("System status"):
        for check in get_startup_checks():
            icon = "✅" if check.ok else ("⚠️" if not check.required else "❌")
            st.markdown(f"{icon} **{check.name}**")
            st.caption(check.message)
    st.divider()
    try:
        conversation_store = get_conversation_store()
        active_conversation = render_conversation_sidebar(conversation_store)
    except Exception as error:
        st.error(f"Conversation storage is unavailable: {error}")

if conversation_store is None or active_conversation is None:
    st.error("The app cannot start because conversation storage is unavailable.")
    st.stop()

startup_checks = get_startup_checks()
storage_failure = next(
    (check for check in startup_checks if check.name == "Storage" and not check.ok),
    None,
)
if storage_failure:
    st.error(f"The app cannot start because local storage is unavailable: {storage_failure.message}")
    st.stop()

st.markdown(
    """
    <div class="workspace-head">
        <div class="workspace-kicker">Private knowledge workspace · Version 4</div>
        <div class="workspace-title-row">
            <h1>Ask your documents.</h1>
            <span class="local-mark">Local by design</span>
        </div>
        <p>Add a document or approved webpage, choose the sources that matter,
        and continue a conversation grounded in your own material.</p>
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
newly_added_source_keys: list[str] = []

if st.session_state.get("source_selection_ui_version") != SOURCE_SELECTION_UI_VERSION:
    st.session_state[SOURCE_SELECTION_KEY] = []
    st.session_state[CONVERSATION_SOURCE_OWNER_KEY] = None
    st.session_state["source_selection_ui_version"] = SOURCE_SELECTION_UI_VERSION

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
                source_key = pdf_source_key(uploaded_file.name)
                store.replace_source(source_key, embedded_pdf_chunks)
                newly_added_source_keys.append(source_key)
            st.success("PDF saved and selected for your next question.")
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
                source_key = website_source_key(website_url)
                store.replace_source(source_key, embedded_website_chunks)
                newly_added_source_keys.append(source_key)
                st.success("Website saved and selected for your next question.")
            except WebsiteLoadError as error:
                st.error(f"I could not add that website: {error}")
            except Exception as error:
                st.error(f"I could not index that website: {error}")

stored_sources = store.list_sources()
available_source_keys = [source.source_key for source in stored_sources]
source_labels = {
    source.source_key: (
        f"{'Website' if source.root_url else 'PDF'} · {source.source_name}"
    )
    for source in stored_sources
}
active_conversation = conversation_store.get(active_conversation.conversation_id)
assert active_conversation is not None
st.session_state[SOURCE_SELECTION_KEY] = reconcile_conversation_sources(
    conversation_id=active_conversation.conversation_id,
    loaded_conversation_id=st.session_state.get(CONVERSATION_SOURCE_OWNER_KEY),
    saved_keys=active_conversation.selected_sources,
    session_keys=st.session_state.get(SOURCE_SELECTION_KEY, []),
    available_keys=available_source_keys,
    newly_added_keys=newly_added_source_keys,
)
st.session_state[CONVERSATION_SOURCE_OWNER_KEY] = (
    active_conversation.conversation_id
)
if tuple(st.session_state[SOURCE_SELECTION_KEY]) != active_conversation.selected_sources:
    active_conversation = conversation_store.set_selected_sources(
        active_conversation.conversation_id,
        tuple(st.session_state[SOURCE_SELECTION_KEY]),
    )

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
    render_conversation_timeline(conversation_store, active_conversation)
    render_history("04")
    st.stop()

st.success("Your documents are ready for questions.", icon="✅")

st.markdown('<div class="section-label">02 · Source scope</div>', unsafe_allow_html=True)
st.subheader("Choose what this conversation can search")

selected_count = len(st.session_state[SOURCE_SELECTION_KEY])
st.markdown(
    f'<div class="source-summary"><span class="source-dot"></span>'
    f'<strong>{selected_count}</strong> of {len(available_source_keys)} library '
    f'{"source" if len(available_source_keys) == 1 else "sources"} selected</div>',
    unsafe_allow_html=True,
)

with st.popover("Choose from document library", use_container_width=True):
    st.caption(
        "Previously uploaded documents stay here, out of the way. "
        "New uploads are selected automatically."
    )
    library_all_col, library_clear_col = st.columns(2)
    if library_all_col.button(
        "Use all", key="library-select-all", use_container_width=True
    ):
        st.session_state[SOURCE_SELECTION_KEY] = available_source_keys
        conversation_store.set_selected_sources(
            active_conversation.conversation_id,
            tuple(available_source_keys),
        )
        st.rerun()
    if library_clear_col.button(
        "Clear", key="library-clear", use_container_width=True
    ):
        st.session_state[SOURCE_SELECTION_KEY] = []
        conversation_store.set_selected_sources(
            active_conversation.conversation_id,
            (),
        )
        st.rerun()
    selected_source_keys = st.multiselect(
        "Documents available for this conversation",
        options=available_source_keys,
        format_func=lambda key: source_labels[key],
        key=SOURCE_SELECTION_KEY,
        help="Only selected documents can contribute to the answer.",
    )
if tuple(selected_source_keys) != active_conversation.selected_sources:
    active_conversation = conversation_store.set_selected_sources(
        active_conversation.conversation_id,
        tuple(selected_source_keys),
    )
selected_source_names = tuple(
    source_labels[key] for key in selected_source_keys
)
if not selected_source_names:
    st.warning("Select at least one source before asking a question.")

searchable_chunks = filter_chunks_by_source_keys(
    embedded_chunks, set(selected_source_keys)
)

render_conversation_timeline(conversation_store, active_conversation)
question = st.chat_input(
    "Ask a question about this conversation's sources",
    disabled=not searchable_chunks,
)

if question:
    try:
        conversation_store.add_message(
            active_conversation.conversation_id,
            "user",
            question,
            original_question=question,
            standalone_query=question,
        )
        if active_conversation.title == "New conversation":
            active_conversation = conversation_store.rename(
                active_conversation.conversation_id,
                conversation_title(question),
            )
    except Exception as error:
        st.error(f"The question could not be saved: {error}")
        st.stop()

    try:
        results = RerankedRetriever(
            searchable_chunks, get_embedding_model()
        ).search(question)
    except ValueError as error:
        conversation_store.add_message(
            active_conversation.conversation_id,
            "assistant",
            f"I could not search this question: {error}",
            status="failed",
        )
        st.rerun()
    else:
        with st.spinner(
            f"{OLLAMA_MODEL} is writing an evidence-grounded answer..."
        ):
            try:
                answer = generate_grounded_answer(question, results)
            except (OllamaError, ValueError) as error:
                conversation_store.add_message(
                    active_conversation.conversation_id,
                    "assistant",
                    f"I could not generate an answer: {error}",
                    status="failed",
                )
                st.rerun()
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
                try:
                    conversation_store.add_message(
                        active_conversation.conversation_id,
                        "assistant",
                        answer,
                        citations=message_citations(answer, results),
                    )
                except Exception as error:
                    st.error(f"The answer was created but could not be saved: {error}")
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        render_answer_sources(answer, results)
                else:
                    st.rerun()

render_history("04")
