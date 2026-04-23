import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from agent.graph import build_graph  # noqa: E402  (after dotenv)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoStream AI Assistant",
    page_icon="🎬",
    layout="centered",
)

# ── Session state init ────────────────────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎬 AutoStream")
    st.caption("AI Sales Assistant")
    st.divider()

    st.subheader("Session Info")
    state = st.session_state.agent_state

    intent_label = state.get("intent", "—")
    st.metric("Detected Intent", intent_label if intent_label else "—")

    st.divider()
    st.subheader("Lead Info")

    col1, col2 = st.columns(2)
    col1.write("**Name**")
    col2.write(state.get("lead_name") or "—")
    col1.write("**Email**")
    col2.write(state.get("lead_email") or "—")
    col1.write("**Platform**")
    col2.write(state.get("lead_platform") or "—")

    if state.get("lead_captured"):
        st.success("✅ Lead Captured!")

    st.divider()
    if st.button("🔄 New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.agent_state = {}
        st.rerun()

# ── Main chat UI ──────────────────────────────────────────────────────────────

st.title("AutoStream AI Assistant")
st.caption("Ask me about pricing, features, or get started today!")

WELCOME = (
    "Hi there! 👋 I'm AutoStream's AI assistant. "
    "I can help you with pricing, features, and getting you set up. "
    "What can I help you with today?"
)

if not st.session_state.chat_history:
    st.session_state.chat_history.append({"role": "assistant", "content": WELCOME})

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message…"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = st.session_state.graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )

        ai_text = result["messages"][-1].content
        st.markdown(ai_text)

    st.session_state.chat_history.append({"role": "assistant", "content": ai_text})
    st.session_state.agent_state = {
        "intent": result.get("intent"),
        "lead_name": result.get("lead_name"),
        "lead_email": result.get("lead_email"),
        "lead_platform": result.get("lead_platform"),
        "lead_captured": result.get("lead_captured", False),
    }

    if result.get("lead_captured"):
        st.balloons()

    st.rerun()
