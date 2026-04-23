import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.rag import RAGRetriever
from agent.tools import mock_lead_capture

_KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"

_llm = None
_retriever = None


def _get_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            temperature=0,
        )
    return _llm


def _get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(str(_KB_PATH))
    return _retriever


SYSTEM_PROMPT = (
    "You are the AutoStream AI assistant. AutoStream is a SaaS platform that provides "
    "automated video editing tools for content creators. Be helpful, concise, and friendly."
)

INTENT_PROMPT = """Classify the user's intent into exactly one of these three categories:

- greeting: casual hello, small talk, or general opener with no product question
- product_inquiry: asking about features, pricing, plans, policies, or how AutoStream works
- high_intent: expressing desire to sign up, try, purchase, or showing strong buying interest

Previous assistant message (for context): "{previous}"
User message: "{message}"

If the assistant just offered to set up an account and the user responds affirmatively (e.g. yes, sure, ok, let's go, sounds good), classify as high_intent.

Reply with a single word only: greeting, product_inquiry, or high_intent"""


# ── Router ────────────────────────────────────────────────────────────────────

def entry_router(state: AgentState) -> str:
    if state.get("awaiting_field"):
        return "collect_field"
    return "detect_intent"


def intent_router(state: AgentState) -> str:
    return state.get("intent", "greeting")


# ── Nodes ─────────────────────────────────────────────────────────────────────

def detect_intent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1].content
    previous_message = messages[-2].content if len(messages) >= 2 else ""
    prompt = INTENT_PROMPT.format(message=last_message, previous=previous_message)
    response = _get_llm().invoke([SystemMessage(content=prompt)])
    raw = response.content.strip().lower()

    if "high_intent" in raw or "high intent" in raw:
        intent = "high_intent"
    elif "product" in raw or "inquiry" in raw:
        intent = "product_inquiry"
    else:
        intent = "greeting"

    return {"intent": intent}


GREET_SYSTEM = (
    f"{SYSTEM_PROMPT}\n\n"
    "Your only job here is to respond warmly to the user's greeting or small talk. "
    "Do NOT ask for any personal details, do NOT describe signup steps, do NOT list plans or pricing. "
    "Just greet them and invite them to ask about AutoStream's features or pricing."
)


def greet_node(state: AgentState) -> AgentState:
    response = _get_llm().invoke([SystemMessage(content=GREET_SYSTEM), *state["messages"]])
    return {"messages": [AIMessage(content=response.content)]}


def rag_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    docs = _get_retriever().retrieve(query)

    CTA = (
        "\n\n---\nWould you like to get started with AutoStream? "
        "I can set up your account right now!"
    )

    if docs:
        context = "\n\n".join(docs)
        system = (
            f"{SYSTEM_PROMPT}\n\n"
            "Answer the user's question using ONLY the knowledge base excerpts provided below.\n"
            "STRICT RULES — you must follow these exactly:\n"
            "1. Only state facts that are explicitly written in the excerpts.\n"
            "2. Do NOT add, infer, or invent any plans, features, pricing, storage limits, "
            "trials, services, or policies that are not word-for-word in the excerpts.\n"
            "3. If the user asks about anything not covered (e.g. Enterprise plan, free trial, "
            "storage, custom templates, extra services), respond with: "
            "'I don't have information about that. Please contact our support team for details.'\n"
            "4. Never use your training knowledge to fill gaps — only the excerpts below count.\n\n"
            f"KNOWLEDGE BASE EXCERPTS:\n{context}"
        )
    else:
        system = (
            f"{SYSTEM_PROMPT}\n\n"
            "You do not have information about that topic in your knowledge base. "
            "Tell the user politely and suggest they contact the support team. "
            "Do NOT make up any details."
        )

    response = _get_llm().invoke([SystemMessage(content=system), *state["messages"]])
    return {"messages": [AIMessage(content=response.content + CTA)]}


def start_collection_node(state: AgentState) -> AgentState:
    reply = (
        "That's awesome — let's get you started! "
        "I just need a few quick details.\n\nWhat's your name?"
    )
    return {
        "messages": [AIMessage(content=reply)],
        "awaiting_field": "name",
    }


def collect_field_node(state: AgentState) -> AgentState:
    field = state.get("awaiting_field")
    value = state["messages"][-1].content.strip()

    updates: dict = {}

    if field == "name":
        updates["lead_name"] = value
        updates["awaiting_field"] = "email"
        updates["messages"] = [
            AIMessage(content=f"Great to meet you, {value}! What's your email address?")
        ]

    elif field == "email":
        updates["lead_email"] = value
        updates["awaiting_field"] = "platform"
        updates["messages"] = [
            AIMessage(
                content="Perfect! Which creator platform do you mainly use? "
                "(e.g., YouTube, Instagram, TikTok, Twitch)"
            )
        ]

    elif field == "platform":
        name = state.get("lead_name", "")
        email = state.get("lead_email", "")
        mock_lead_capture(name, email, value)

        updates["lead_platform"] = value
        updates["awaiting_field"] = None
        updates["lead_captured"] = True
        updates["messages"] = [
            AIMessage(
                content=(
                    f"You're all set, {name}! 🎉\n\n"
                    f"We'll reach out to **{email}** shortly to activate your AutoStream account. "
                    f"Welcome to the {value} creator community — can't wait to see what you create!"
                )
            )
        ]

    return updates


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("detect_intent", detect_intent_node)
    builder.add_node("greet", greet_node)
    builder.add_node("rag_respond", rag_node)
    builder.add_node("start_collection", start_collection_node)
    builder.add_node("collect_field", collect_field_node)

    builder.add_conditional_edges(
        START,
        entry_router,
        {"detect_intent": "detect_intent", "collect_field": "collect_field"},
    )

    builder.add_conditional_edges(
        "detect_intent",
        intent_router,
        {
            "greeting": "greet",
            "product_inquiry": "rag_respond",
            "high_intent": "start_collection",
        },
    )

    builder.add_edge("greet", END)
    builder.add_edge("rag_respond", END)
    builder.add_edge("start_collection", END)
    builder.add_edge("collect_field", END)

    return builder.compile(checkpointer=MemorySaver())
