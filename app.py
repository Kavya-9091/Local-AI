import html
import hashlib
import json
import time
import uuid
from io import BytesIO
from pathlib import Path

import requests
import speech_recognition as sr
import streamlit as st


APP_DIR = Path(__file__).parent
HISTORY_FILE = APP_DIR / "chat_history.json"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Explain concepts clearly and simply. "
    "When useful, respond with examples."
)
MODELS = ["llama3.2", "qwen2.5:3b", "llama3.1", "llama3", "mistral", "gemma2", "qwen2.5"]
SUGGESTIONS = [
    "Explain a Generative AI concept simply",
    "Write Python code and walk me through it",
    "Draft a short lesson plan",
    "Help debug an error message",
]


st.set_page_config(
    page_title="Local AI",
    page_icon=".",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root {
        --surface: #ffffff;
        --sidebar: #f7f7f8;
        --sidebar-hover: #ececec;
        --text: #171717;
        --muted: #6b7280;
        --border: #e5e7eb;
        --user: #f4f4f4;
    }

    .stApp { background: var(--surface); color: var(--text); }
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    #MainMenu,
    footer { display: none; }

    section[data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--border);
        min-width: 280px;
    }

    section[data-testid="stSidebar"] > div { padding: 18px 12px 14px; }
    .main .block-container { max-width: 920px; padding: 28px 32px 120px; }

    .brand-row {
        padding: 2px 4px 16px;
        font-size: 22px;
        font-weight: 700;
    }

    .sidebar-title {
        color: #8a8f98;
        font-size: 13px;
        font-weight: 600;
        margin: 18px 6px 8px;
    }

    .empty-state {
        min-height: 48vh;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .empty-title {
        font-size: 32px;
        line-height: 1.2;
        font-weight: 500;
        color: #202123;
    }

    .suggestions-row {
        max-width: 720px;
        margin: 0 auto;
    }

    .suggestions-row .stButton > button {
        min-height: 58px;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: #ffffff;
        color: #4b5563;
        padding: 14px 16px;
        text-align: left;
        justify-content: flex-start;
        font-size: 14px;
        font-weight: 400;
    }

    .suggestions-row .stButton > button:hover {
        background: #f9fafb;
        border-color: #d1d5db;
    }

    .chat-wrap {
        display: flex;
        flex-direction: column;
        gap: 20px;
        padding-top: 18px;
    }

    .message-row {
        display: flex;
        width: 100%;
        margin: 18px 0;
    }

    .message-row.user { justify-content: flex-end; }
    .message-row.assistant { justify-content: flex-start; }

    .message-bubble {
        max-width: min(760px, 82%);
        border-radius: 22px;
        padding: 12px 16px;
        line-height: 1.65;
        font-size: 16px;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .message-row.user .message-bubble {
        background: var(--user);
        color: var(--text);
    }

    .message-row.assistant .message-bubble {
        background: transparent;
        color: var(--text);
        padding-left: 0;
    }

    .edit-beside-bubble {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        margin-top: 8px;
        margin-bottom: 18px;
        padding-right: 0;
        transform: none;
    }

    .edit-beside-bubble .stButton > button {
        width: 38px;
        height: 38px;
        min-height: 38px;
        padding: 0;
        border-radius: 12px;
        color: var(--muted);
        background: #f4f4f4;
        font-size: 17px;
        justify-content: center;
        border: 1px solid #e5e7eb;
        box-shadow: none;
    }

    .edit-beside-bubble .stButton > button:hover {
        background: #e9e9e9;
        color: #171717;
    }

    .attachment-note {
        margin-top: -12px;
        color: var(--muted);
        font-size: 13px;
        text-align: right;
    }

    div[data-testid="stChatInput"] {
        max-width: 760px;
        margin: 0 auto 20px;
        left: 50%;
        transform: translateX(-50%);
        background: #ffffff;
        border: 1px solid #d9d9d9;
        border-radius: 28px;
        box-shadow: 0 10px 32px rgba(0,0,0,0.10);
        padding: 0;
        overflow: hidden;
    }

    div[data-testid="stChatInput"] > div {
        background: transparent;
        border: 0;
        box-shadow: none;
    }

    div[data-testid="stChatInput"] textarea {
        min-height: 54px;
        max-height: 160px;
        border-radius: 0;
        border: 0;
        background: transparent;
        box-shadow: none;
        padding: 15px 56px 15px 270px;
        font-size: 16px;
        line-height: 1.45;
    }

    div[data-testid="stChatInput"] textarea:focus {
        border: 0;
        box-shadow: none;
        outline: none;
    }

    div[data-testid="stChatInputSubmitButton"] button {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #171717;
        color: #ffffff;
        margin-right: 8px;
    }

    div[data-testid="stChatInputFileUploader"] button,
    div[data-testid="stChatInput"] button[aria-label*="Attach"],
    div[data-testid="stChatInput"] button[title*="Attach"] {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: transparent;
        border: 0;
        color: #171717;
        margin-left: 8px;
        box-shadow: none;
    }

    div[data-testid="stChatInput"] button {
        align-self: center;
    }

    .composer-tools {
        display: none;
    }

    .composer-tools [data-testid="stSelectbox"] {
        width: 150px;
        margin: 0;
    }

    .composer-tools [data-testid="stSelectbox"] label {
        display: none;
    }

    .composer-tools [data-baseweb="select"] > div {
        min-height: 32px;
        border-radius: 999px;
        border: 0;
        background: #f4f4f4;
        font-size: 13px;
        box-shadow: none;
    }

    .composer-tools [data-testid="stAudioInput"] {
        width: 42px;
        margin: 0;
    }

    .composer-tools [data-testid="stAudioInput"] label,
    .composer-tools [data-testid="stAudioInput"] section {
        display: none;
    }

    .composer-tools [data-testid="stAudioInput"] button {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        border: 0;
        background: #f4f4f4;
        color: #171717;
        padding: 0;
    }

    .composer-tools div[data-testid="column"] {
        padding: 0;
    }

    div[data-testid="stForm"] {
        max-width: 960px;
        margin: 26px auto 18px;
        background: #ffffff;
        border: 1px solid #d9d9d9;
        border-radius: 26px;
        box-shadow: 0 10px 32px rgba(0,0,0,0.10);
        padding: 14px 18px 10px;
    }

    div[data-testid="stForm"] > div {
        gap: 0;
    }

    div[data-testid="stForm"] [data-testid="column"] {
        display: flex;
        align-items: center;
    }

    div[data-testid="stForm"] textarea,
    div[data-testid="stForm"] input {
        border: 0;
        box-shadow: none;
        font-size: 16px;
        background: transparent;
    }

    div[data-testid="stForm"] textarea {
        min-height: 70px;
        resize: none;
        padding: 8px 0 10px;
        line-height: 1.5;
        outline: none;
        border-color: transparent;
    }

    div[data-testid="stForm"] textarea:focus,
    div[data-testid="stForm"] textarea:focus-visible,
    div[data-testid="stForm"] textarea:focus-within {
        border: 0;
        outline: none;
        box-shadow: none;
    }

    div[data-testid="stTextArea"] > div,
    div[data-testid="stTextArea"] > div > div {
        border: 0;
        outline: none;
        box-shadow: none;
        background: transparent;
    }

    div[data-testid="stTextArea"] small,
    div[data-testid="InputInstructions"] {
        display: none;
    }

    div[data-testid="stForm"] div[data-baseweb="select"] > div {
        border: 0;
        background: #f4f4f4;
        border-radius: 999px;
        min-height: 38px;
        font-size: 13px;
        min-width: 170px;
    }

    div[data-testid="stForm"] label {
        display: none;
    }

    div[data-testid="stForm"] .stButton > button,
    div[data-testid="stForm"] button {
        min-height: 38px;
        border-radius: 999px;
    }

    div[data-testid="stFormSubmitButton"] button {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: #171717;
        color: #ffffff;
        padding: 0;
        justify-content: center;
        font-size: 18px;
    }

    div[data-testid="stFileUploader"] {
        width: 42px;
        height: 42px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        display: none;
    }

    div[data-testid="stFileUploader"] section {
        width: 42px;
        height: 42px;
        min-height: 42px;
        padding: 0;
        border: 0;
        background: transparent;
    }

    div[data-testid="stFileUploader"] section > div {
        display: none;
    }

    div[data-testid="stFileUploader"] button {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        padding: 0;
        font-size: 0;
        background: transparent;
        color: transparent;
        border: 0;
        position: relative;
        overflow: hidden;
        min-height: 38px;
    }

    div[data-testid="stFileUploader"] button::after {
        content: "+";
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        line-height: 1;
        font-weight: 300;
        color: #171717;
    }

    div[data-testid="stAudioInput"] {
        width: 42px;
        height: 42px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    div[data-testid="stAudioInput"] label,
    div[data-testid="stAudioInput"] section,
    div[data-testid="stAudioInput"] audio,
    div[data-testid="stAudioInput"] [data-testid="stAudioInputWaveSurfer"],
    div[data-testid="stAudioInput"] [data-testid="stAudioInputRecordedAudio"] {
        display: none;
    }

    div[data-testid="stAudioInput"] > div {
        width: 42px;
        height: 42px;
        padding: 0;
        margin: 0;
        background: transparent;
        border: 0;
        box-shadow: none;
    }

    div[data-testid="stAudioInput"] button {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        padding: 0;
        background: transparent;
        color: #171717;
        border: 0;
        box-shadow: none;
        min-height: 38px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 0;
        background: transparent;
        color: #202123;
        justify-content: flex-start;
        font-weight: 500;
        padding: 9px 10px;
    }

    .stButton > button:hover {
        background: var(--sidebar-hover);
        color: #111827;
    }

    div[data-testid="stSidebar"] .stButton:first-of-type > button {
        background: #ececec;
    }

    @media (max-width: 760px) {
        .main .block-container { padding: 18px 16px 110px; }
        .message-bubble { max-width: 92%; }
        div[data-testid="stChatInput"] textarea {
            padding-left: 170px;
        }
        .composer-tools {
            left: 20px;
            bottom: 29px;
            width: 150px;
            transform: none;
        }
        .composer-tools [data-testid="stSelectbox"] {
            width: 108px;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_history() -> dict:
    if not HISTORY_FILE.exists():
        return {"conversations": []}
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return {"conversations": []}
    conversations = data.get("conversations", [])
    if not isinstance(conversations, list):
        conversations = []
    conversations = [
        conversation
        for conversation in conversations
        if conversation.get("messages") or conversation.get("title") != "New chat"
    ]
    return {"conversations": conversations}


def save_history(history: dict) -> None:
    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)


def new_conversation() -> dict:
    now = int(time.time())
    return {
        "id": uuid.uuid4().hex,
        "title": "New chat",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }


def get_conversation(history: dict, conversation_id: str | None) -> dict:
    for conversation in history["conversations"]:
        if conversation["id"] == conversation_id:
            return conversation
    return new_conversation()


def save_conversation_if_needed(history: dict, conversation: dict) -> None:
    if any(item["id"] == conversation["id"] for item in history["conversations"]):
        return
    history["conversations"].insert(0, conversation)
    st.session_state.active_conversation_id = conversation["id"]


def update_title(conversation: dict, prompt: str) -> None:
    if conversation["title"] != "New chat":
        return
    title = " ".join(prompt.strip().split())
    conversation["title"] = title[:42] + ("..." if len(title) > 42 else "")


def check_ollama_running() -> bool:
    try:
        return requests.get(OLLAMA_BASE_URL, timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False


def transcribe_audio(audio_file) -> str:
    recognizer = sr.Recognizer()
    audio_bytes = audio_file.getvalue()
    with sr.AudioFile(BytesIO(audio_bytes)) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)


def render_message(role: str, content: str) -> None:
    safe_content = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
<div class="message-row {role}">
  <div class="message-bubble">{safe_content}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_user_message_with_edit(content: str) -> None:
    safe_content = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
<div class="message-row user user-message-with-edit">
  <div class="message-bubble">{safe_content}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_attachments(files: list[str]) -> None:
    if not files:
        return
    names = ", ".join(html.escape(name) for name in files)
    st.markdown(f'<div class="attachment-note">Attached: {names}</div>', unsafe_allow_html=True)


def move_active_to_top(history: dict, conversation_id: str) -> None:
    history["conversations"].sort(key=lambda item: item.get("updated_at", 0), reverse=True)
    active_index = next(
        (index for index, item in enumerate(history["conversations"]) if item["id"] == conversation_id),
        None,
    )
    if active_index is not None:
        history["conversations"].insert(0, history["conversations"].pop(active_index))


if "history" not in st.session_state:
    st.session_state.history = load_history()

if "active_conversation_id" not in st.session_state:
    conversations = st.session_state.history["conversations"]
    st.session_state.active_conversation_id = conversations[0]["id"] if conversations else None

with st.sidebar:
    st.markdown('<div class="brand-row">Local AI</div>', unsafe_allow_html=True)

    if st.button("New chat", key="new_chat"):
        st.session_state.editing_message_index = None
        st.session_state.active_conversation_id = None
        st.rerun()

    st.markdown('<div class="sidebar-title">Chats</div>', unsafe_allow_html=True)
    for saved_conversation in st.session_state.history["conversations"]:
        if st.button(saved_conversation["title"], key=f"chat_{saved_conversation['id']}"):
            st.session_state.editing_message_index = None
            st.session_state.active_conversation_id = saved_conversation["id"]
            st.rerun()

    if st.button("Delete current chat", key="delete_chat"):
        if st.session_state.active_conversation_id is not None:
            st.session_state.history["conversations"] = [
                item
                for item in st.session_state.history["conversations"]
                if item["id"] != st.session_state.active_conversation_id
            ]
        conversations = st.session_state.history["conversations"]
        st.session_state.editing_message_index = None
        st.session_state.active_conversation_id = conversations[0]["id"] if conversations else None
        save_history(st.session_state.history)
        st.rerun()


history = st.session_state.history
conversation = get_conversation(history, st.session_state.active_conversation_id)

if not check_ollama_running():
    st.error("Ollama is not running. Open Ollama or run `ollama serve`, then refresh this page.")
    st.stop()

messages = conversation["messages"]

if not messages:
    st.markdown(
        """
<div class="empty-state">
  <div class="empty-title">What can I help with?</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="suggestions-row">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for index, suggestion in enumerate(SUGGESTIONS):
        column = col1 if index % 2 == 0 else col2
        if column.button(suggestion, key=f"suggestion_{index}"):
            st.session_state.pending_prompt = suggestion
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for index, message in enumerate(messages):
        if st.session_state.get("editing_message_index") == index:
            with st.form(f"edit_message_{index}"):
                edited_content = st.text_area(
                    "Edit message",
                    value=message["content"],
                    label_visibility="collapsed",
                    height=110,
                )
                save_col, cancel_col, spacer_col = st.columns([0.8, 0.8, 4])
                save_edit = save_col.form_submit_button("Save")
                cancel_edit = cancel_col.form_submit_button("Cancel")
            if save_edit:
                messages[index]["content"] = edited_content.strip()
                del messages[index + 1 :]
                conversation["updated_at"] = int(time.time())
                save_history(history)
                st.session_state.editing_message_index = None
                st.rerun()
            if cancel_edit:
                st.session_state.editing_message_index = None
                st.rerun()
            continue

        if message["role"] == "user":
            left_col, bubble_col, edit_col = st.columns([3, 8, 0.55])
            with bubble_col:
                render_message(message["role"], message["content"])
                render_attachments(message.get("attachments", []))
            with edit_col:
                st.markdown('<div class="edit-beside-bubble">', unsafe_allow_html=True)
                if st.button("✎", key=f"edit_{conversation['id']}_{index}", help="Edit message"):
                    st.session_state.editing_message_index = index
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            render_message(message["role"], message["content"])
            render_attachments(message.get("attachments", []))
    st.markdown("</div>", unsafe_allow_html=True)

live_response_area = st.container()

temperature = 0.7
system_prompt = DEFAULT_SYSTEM_PROMPT

user_prompt = st.session_state.pop("pending_prompt", None)
uploaded_files = []
with st.form("composer", clear_on_submit=True):
    typed_prompt = st.text_area(
        "Message",
        placeholder="Ask anything",
        label_visibility="collapsed",
        height=74,
    )
    upload_col, mic_col, spacer_col, model_col, send_col = st.columns([0.38, 0.42, 5.0, 1.55, 0.42])
    with upload_col:
        uploaded_image = st.file_uploader(
            "+",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )
    with mic_col:
        voice_file = st.audio_input("Voice", label_visibility="collapsed")
    with model_col:
        model = st.selectbox("Model", MODELS, index=0, label_visibility="collapsed")
    with send_col:
        button_label = "Stop" if st.session_state.get("is_generating") else "↑"
        submitted = st.form_submit_button(button_label)

if uploaded_image is not None:
    uploaded_files = [uploaded_image.name]

if submitted:
    user_prompt = typed_prompt
    if voice_file is not None and not user_prompt:
        voice_bytes = voice_file.getvalue()
        voice_hash = hashlib.sha256(voice_bytes).hexdigest()
        try:
            if st.session_state.get("last_voice_hash") != voice_hash:
                user_prompt = transcribe_audio(voice_file)
                st.session_state.last_voice_hash = voice_hash
                st.info(f"Voice typed: {user_prompt}")
        except sr.UnknownValueError:
            st.warning("I could not understand the recording. Please try again.")
        except sr.RequestError:
            st.warning("Speech transcription needs internet access. Please type the message.")
        except Exception as error:
            st.warning(f"Could not transcribe voice: {error}")

if submitted and (user_prompt or uploaded_files):
    st.session_state.is_generating = True
    prompt_text = user_prompt or "Please respond to the attached file."
    save_conversation_if_needed(history, conversation)
    update_title(conversation, prompt_text)
    conversation["messages"].append(
        {"role": "user", "content": prompt_text, "attachments": uploaded_files}
    )
    conversation["updated_at"] = int(time.time())
    save_history(history)

    with live_response_area:
        render_message("user", prompt_text)
        render_attachments(uploaded_files)

    messages_for_ollama = [{"role": "system", "content": system_prompt}]
    messages_for_ollama.extend(
        {"role": message["role"], "content": message["content"]}
        for message in conversation["messages"]
    )

    response_placeholder = live_response_area.empty()
    full_response = ""

    try:
        payload = {
            "model": model,
            "messages": messages_for_ollama,
            "stream": True,
            "options": {"temperature": temperature},
        }

        with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=180) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_response += content
                    safe_response = html.escape(full_response + "|").replace("\n", "<br>")
                    response_placeholder.markdown(
                        f"""
<div class="message-row assistant">
  <div class="message-bubble">{safe_response}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                if chunk.get("done", False):
                    break

        response_placeholder.empty()
        with live_response_area:
            render_message("assistant", full_response)

    except requests.exceptions.HTTPError as error:
        full_response = (
            f"HTTP Error: {error}\n\n"
            f"Most likely the model '{model}' is not downloaded.\n\n"
            f"Run this command in terminal:\n\n"
            f"ollama pull {model}"
        )
        st.session_state.is_generating = False
        with live_response_area:
            response_placeholder.error(full_response)
    except Exception as error:
        full_response = f"Error: {error}"
        st.session_state.is_generating = False
        with live_response_area:
            response_placeholder.error(full_response)

    conversation["messages"].append({"role": "assistant", "content": full_response})
    conversation["updated_at"] = int(time.time())
    move_active_to_top(history, conversation["id"])
    save_history(history)
    st.session_state.is_generating = False
    st.rerun()

