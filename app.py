import os
import streamlit as st
from dotenv import load_dotenv

# ── Load secrets ──────────────────────────────────────────────────────────────
load_dotenv()

# Streamlit Cloud stores secrets differently than .env
HF_TOKEN = (
    st.secrets.get("HF_TOKEN")
    or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_TOKEN")
)
if not HF_TOKEN:
    st.error("❌ HF_TOKEN not found. Add it to Streamlit Cloud → App Settings → Secrets.")
    st.stop()

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ask Sikandar",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg-deep:      #0a0d13;
    --bg-card:      #111520;
    --bg-input:     #161b28;
    --border:       #1e2740;
    --accent:       #4f7cff;
    --accent-glow:  #4f7cff33;
    --accent-warm:  #ff9f43;
    --text-primary: #e8ecf4;
    --text-muted:   #7a849e;
    --user-bubble:  #1a2540;
    --ai-bubble:    #0f1623;
    --radius:       14px;
}

/* ── App background ── */
.stApp {
    background: var(--bg-deep) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent {
    background: var(--user-bubble) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stChatMessageContent {
    background: var(--ai-bubble) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}

.stChatMessageContent p {
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50px !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    background: transparent !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

/* ── Sidebar avatar ring ── */
.avatar-ring {
    width: 88px; height: 88px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-warm));
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1rem;
    font-size: 2.2rem;
}

/* ── Social link pills ── */
.link-pill {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid var(--border);
    color: var(--text-muted) !important;
    text-decoration: none !important;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px 3px 0 0;
    transition: all 0.2s;
    background: var(--bg-deep);
}
.link-pill:hover {
    border-color: var(--accent);
    color: var(--accent) !important;
    background: var(--accent-glow);
}

/* ── Starter prompt buttons ── */
.stButton > button {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    padding: 0.6rem 0.8rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.2s !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--text-primary) !important;
    background: var(--accent-glow) !important;
}

/* ── Stat cards in sidebar ── */
.stat-row {
    display: flex; gap: 8px; margin: 12px 0;
}
.stat-card {
    flex: 1;
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 8px;
    text-align: center;
}
.stat-num { font-size: 1.1rem; font-weight: 600; color: var(--accent); }
.stat-label { font-size: 0.68rem; color: var(--text-muted); margin-top: 2px; }

/* ── Header title ── */
.chat-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--text-primary);
    margin: 0 0 4px;
}
.chat-sub {
    font-size: 0.83rem;
    color: var(--text-muted);
    margin-bottom: 1.2rem;
}

/* ── Thinking dots ── */
.thinking {
    display: flex; gap: 5px; align-items: center;
    padding: 12px 16px;
}
.dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent);
    animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-6px); opacity: 1; }
}

/* ── Suggestion label ── */
.suggest-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE  (cached so it loads only once per session)
# ══════════════════════════════════════════════════════════════════════════════

MAX_HISTORY_TURNS = 6

class CappedChatMessageHistory(ChatMessageHistory):
    """Keeps only the last N exchanges to prevent context window overflow."""
    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]


@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=768,
        temperature=0.2,
        repetition_penalty=1.1,
        huggingfacehub_api_token=HF_TOKEN,
        timeout=300,
    )
    chat_model = ChatHuggingFace(llm=llm)

    system_instruction = """You are the personal AI representative for Sikandar Farooq Saani.
Answer questions about his background, projects, skills, family, lifestyle, and philosophy \
using ONLY the retrieved context below.

CORE FACTS (always accurate — use these directly):
- Hometown: Mansehra, Pakistan. Current city: Lahore, Pakistan.
- GitHub:    https://github.com/SikandarFarooqSaani
- Kaggle:    https://www.kaggle.com/sikandarfarooqsaani
- Website:   https://sfs.surge.sh
- LinkedIn:  https://www.linkedin.com/in/sikandar-farooq-8b1154224/
- Instagram: https://www.instagram.com/sani_sikandar/
- Hugging Face: https://huggingface.co/SikandarFarooqSaani

STRICT RULES:
1. Never hedge with "Based on context", "It appears", or "From what I understand". Be direct.
2. Always refer to Sikandar in the third person: He / His / Sikandar.
3. List ALL relevant items when multiple appear (projects, skills, etc.).
4. For social links or location, use CORE FACTS above.
5. If genuinely not in context, say: "I don't have that specific information about Sikandar yet."
6. Never hallucinate or invent details.
7. Be concise and professional — no unnecessary filler.

Retrieved Context:
{context}"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        source_labels = {
            "biography":        "PERSONAL BIOGRAPHY",
            "project_cards":    "PROJECT DETAILS",
            "skills_profile":   "SKILLS & TECH STACK",
            "github_portfolio": "GITHUB REPOSITORY",
            "github_profile":   "GITHUB PROFILE",
            "website_portfolio":"PERSONAL WEBSITE",
            "kaggle_profile":   "KAGGLE PROFILE",
        }
        parts = []
        for doc in docs:
            label = source_labels.get(doc.metadata.get("source", ""), "CONTEXT")
            parts.append(f"[{label}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    context_chain = (lambda x: x["input"]) | retriever | format_docs

    core_runnable = (
        RunnablePassthrough.assign(context=context_chain)
        | prompt_template
        | chat_model
        | StrOutputParser()
    )

    history_store: dict[str, CappedChatMessageHistory] = {}

    def get_history(session_id: str) -> CappedChatMessageHistory:
        if session_id not in history_store:
            history_store[session_id] = CappedChatMessageHistory()
        return history_store[session_id]

    chain = RunnableWithMessageHistory(
        core_runnable,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Avatar
    st.markdown('<div class="avatar-ring">🧑‍💻</div>', unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center'>"
        "<div style='font-family:DM Serif Display,serif;font-size:1.25rem;"
        "color:#e8ecf4;font-weight:400'>Sikandar Farooq Saani</div>"
        "<div style='font-size:0.78rem;color:#7a849e;margin-top:3px'>"
        "AI/ML Engineer · Lahore, PK</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Stats
    st.markdown("""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-num">4+</div>
            <div class="stat-label">Projects</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">23</div>
            <div class="stat-label">Age</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">6d</div>
            <div class="stat-label">Gym / wk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Social links
    st.markdown(
        "<div style='font-size:0.75rem;color:#7a849e;text-transform:uppercase;"
        "letter-spacing:0.08em;font-weight:600;margin-bottom:8px'>Connect</div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div>
        <a class="link-pill" href="https://github.com/SikandarFarooqSaani" target="_blank">⚙ GitHub</a>
        <a class="link-pill" href="https://www.kaggle.com/sikandarfarooqsaani" target="_blank">📊 Kaggle</a>
        <a class="link-pill" href="https://sfs.surge.sh" target="_blank">🌐 Website</a>
        <a class="link-pill" href="https://www.linkedin.com/in/sikandar-farooq-8b1154224/" target="_blank">💼 LinkedIn</a>
        <a class="link-pill" href="https://huggingface.co/SikandarFarooqSaani" target="_blank">🤗 HuggingFace</a>
        <a class="link-pill" href="https://www.instagram.com/sani_sikandar/" target="_blank">📸 Instagram</a>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Core skills tags
    st.markdown(
        "<div style='font-size:0.75rem;color:#7a849e;text-transform:uppercase;"
        "letter-spacing:0.08em;font-weight:600;margin-bottom:8px'>Core Skills</div>",
        unsafe_allow_html=True,
    )
    skills = ["PyTorch", "LangChain", "RAG", "NLP", "YOLOv8", "BERT",
              "ChromaDB", "Streamlit", "XGBoost", "LLMs"]
    pills_html = " ".join(
        f"<span style='display:inline-block;background:#0a0d13;border:1px solid #1e2740;"
        f"border-radius:20px;padding:3px 10px;font-size:0.72rem;color:#7a849e;"
        f"margin:2px 2px 0 0'>{s}</span>"
        for s in skills
    )
    st.markdown(pills_html, unsafe_allow_html=True)

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown(
        "<div style='font-size:0.7rem;color:#3a4255;text-align:center;margin-top:8px'>"
        "Powered by Llama 3.1 · LangChain · ChromaDB</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='chat-header'>Ask Sikandar's AI</div>"
    "<div class='chat-sub'>Ask anything about his background, projects, skills, or life.</div>",
    unsafe_allow_html=True,
)

# ── Starter prompt buttons (shown only when chat is empty) ───────────────────
STARTER_PROMPTS = [
    "🛠️  What projects has Sikandar built?",
    "💼  What roles is Sikandar applying for?",
    "🤖  What AI/ML skills does Sikandar have?",
    "📖  What books and poets inspire Sikandar?",
    "🎯  What are Sikandar's goals for the next 5 years?",
    "🔗  Share Sikandar's GitHub and portfolio links.",
]

if not st.session_state.messages:
    st.markdown("<div class='suggest-label'>Suggested questions</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, prompt in enumerate(STARTER_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"starter_{i}"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

# ── Render conversation history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about Sikandar…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Thinking animation while pipeline loads
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            "<div class='thinking'>"
            "<div class='dot'></div><div class='dot'></div><div class='dot'></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            chain = load_rag_pipeline()

            # Stream response token-by-token
            full_response = ""
            response_placeholder = st.empty()
            thinking_placeholder.empty()

            for chunk in chain.stream(
                {"input": user_input},
                config={"configurable": {"session_id": st.session_state.session_id}},
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"⚠️ Something went wrong: `{type(e).__name__}`. Please try again."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
