"""Streamlit chat interface for the ML RAG Assistant.

Run with: streamlit run app.py
"""

import streamlit as st

from api_client import APIError, ask_question, check_health

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML RAG Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }
    .app-header {
        text-align: center;
        margin-bottom: 0.25rem;
        animation: fadeInDown 0.6s ease-out;
    }
    .app-header h1 {
        font-size: 2rem;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        animation: fadeInDown 0.6s ease-out 0.1s both;
    }
    .source-chip {
        display: inline-block;
        background-color: #eef2ff;
        color: #4338ca;
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        margin: 0.15rem 0.3rem 0.15rem 0;
        font-size: 0.8rem;
        border: 1px solid #c7d2fe;
        transition: transform 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease;
        animation: chipPop 0.3s ease-out both;
    }
    .source-chip:hover {
        transform: translateY(-2px);
        background-color: #e0e7ff;
        box-shadow: 0 2px 8px rgba(67, 56, 202, 0.15);
    }
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .status-online {
        background-color: #22c55e;
        box-shadow: 0 0 0 rgba(34, 197, 94, 0.5);
        animation: pulseGreen 2s infinite;
    }
    .status-offline { background-color: #ef4444; }

    /* Chat message entrance animation */
    [data-testid="stChatMessage"] {
        animation: fadeInUp 0.35s ease-out both;
    }

    /* Smooth button interactions */
    .stButton > button {
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes chipPop {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes pulseGreen {
        0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.5); }
        70% { box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: backend status + about
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 Backend Status")
    health = check_health()
    if health:
        st.markdown(
            '<span class="status-dot status-online"></span>**Online**',
            unsafe_allow_html=True,
        )
        st.caption(f"Indexed chunks: {health.get('indexed_chunks', '—')}")
    else:
        st.markdown(
            '<span class="status-dot status-offline"></span>**Offline**',
            unsafe_allow_html=True,
        )
        st.caption("Start the FastAPI backend to enable chat.")

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "This assistant answers Machine Learning questions grounded in course "
        "lecture notes, scikit-learn documentation, and pandas documentation. "
        "Every answer cites its source."
    )
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header"><h1>🤖 ML RAG Assistant</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Ask a Machine Learning question — '
    "answers are grounded in your course materials and documentation.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown(
                "".join(f'<span class="source-chip">📄 {s}</span>' for s in msg["sources"]),
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
question = st.chat_input("Ask about Regression, Classification, Clustering, pandas...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_question(question)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)
                if sources:
                    st.markdown(
                        "".join(f'<span class="source-chip">📄 {s}</span>' for s in sources),
                        unsafe_allow_html=True,
                    )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except APIError as e:
                error_msg = f"⚠️ {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "sources": []}
                )
