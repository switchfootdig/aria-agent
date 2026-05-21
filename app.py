import streamlit as st
import anthropic
import re
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA — Requirements Agent",
    page_icon="⚡",
    layout="centered"
)

# ── Load system prompt ────────────────────────────────────────
@st.cache_resource
def load_system_prompt():
    p = Path("system_prompt.txt")
    return p.read_text() if p.exists() else "You are ARIA, an energy transmission requirements agent."

# ── Helpers ───────────────────────────────────────────────────
def extract_requirements(text):
    m = re.search(r"<requirements>(.*?)</requirements>", text, re.DOTALL)
    if not m:
        return None
    try:
        import json
        return json.loads(m.group(1).strip())
    except Exception:
        return None

def clean_text(text):
    text = re.sub(r"\[PHASE:\d\]\s*", "", text)
    text = re.sub(r"<requirements>.*?</requirements>", "", text, flags=re.DOTALL)
    return text.strip()

def get_phase(text):
    m = re.search(r"\[PHASE:(\d)\]", text)
    return int(m.group(1)) if m else None

PHASES = ["Orientation","Problem","Data Landscape","Consumers","Compliance","Constraints","Validation","Complete"]

# ── Session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "phase" not in st.session_state:
    st.session_state.phase = 0
if "complete" not in st.session_state:
    st.session_state.complete = False
if "requirements" not in st.session_state:
    st.session_state.requirements = None
if "started" not in st.session_state:
    st.session_state.started = False

# ── UI: Header ────────────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    st.image("static/acerez_logo2.png", width=120)
with col2:
    st.markdown("## ⚡ ARIA")
    st.caption("Automated Requirements Intelligence Agent — Energy Transmission")
    
# Phase progress bar
phase = st.session_state.phase
progress = phase / 7
st.progress(progress, text=f"Phase {phase + 1} of 7: {PHASES[phase]}")

st.divider()

# ── Start interview ───────────────────────────────────────────
def call_claude(messages):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=load_system_prompt(),
        messages=messages
    )
    return response.content[0].text

if not st.session_state.started:
    with st.spinner("Starting your session with ARIA..."):
        opening_msg = {"role": "user", "content": "Hello, I'm ready to capture a new data platform use case. Please introduce yourself briefly and begin the interview."}
        reply = call_claude([opening_msg])
        st.session_state.history.append(opening_msg)
        st.session_state.history.append({"role": "assistant", "content": reply})
        p = get_phase(reply)
        if p is not None:
            st.session_state.phase = p
        st.session_state.started = True
        st.rerun()

# ── Chat history ──────────────────────────────────────────────
for msg in st.session_state.history:
    if msg["role"] == "user" and "Please introduce yourself" in msg["content"]:
        continue  # hide the internal starter message
    with st.chat_message("assistant" if msg["role"] == "assistant" else "user",
                         avatar="⚡" if msg["role"] == "assistant" else "👤"):
        st.write(clean_text(msg["content"]))

# ── Requirements card ─────────────────────────────────────────
if st.session_state.complete and st.session_state.requirements:
    req = st.session_state.requirements
    st.success("✅ Requirements document captured!")
    with st.expander("View structured requirements", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Use Case", req.get("use_case_name", "—"))
            st.metric("Priority", req.get("priority", "—"))
            st.metric("NERC CIP", "Applicable" if req.get("nerc_cip") else "Not applicable")
        with col2:
            st.metric("Owner Role", req.get("business_owner_role", "—"))
            st.metric("Deadline", req.get("deadline") or "Not set")
            st.metric("Open Questions", len(req.get("open_questions", [])))

        st.download_button(
            "⬇️ Download Requirements JSON",
            data=__import__("json").dumps(req, indent=2),
            file_name=f"{req.get('use_case_name','requirements').replace(' ','-').lower()}.json",
            mime="application/json"
        )

    if st.button("🔄 Start new use case"):
        for key in ["history","phase","complete","requirements","started"]:
            del st.session_state[key]
        st.rerun()

# ── Chat input ────────────────────────────────────────────────
elif not st.session_state.complete:
    user_input = st.chat_input("Type your response here...")
    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("ARIA is thinking..."):
                reply = call_claude(st.session_state.history)
                st.session_state.history.append({"role": "assistant", "content": reply})

                p = get_phase(reply)
                if p is not None:
                    st.session_state.phase = p

                req = extract_requirements(reply)
                if req:
                    st.session_state.requirements = req
                    st.session_state.complete = True
                    st.session_state.phase = 7

                st.write(clean_text(reply))
                st.rerun()
