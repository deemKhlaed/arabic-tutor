"""Builds the tutor graph and runs the Streamlit UI. Run with: streamlit run Main.py"""

import datetime
import html
import os
import time
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from Database import Database
from model import Models
from Nodes import TutorNodes, TutorState
from tools import (
    UI_TEXT,
    canvas_to_png,
    is_canvas_empty,
    is_rtl,
    learner_language,
    lesson_items,
    letter_with_marks,
)

load_dotenv()

VOWEL_MARKS = {
    "fatha": "◌َ", "kasra": "◌ِ", "damma": "◌ُ", "sukun": "◌ْ",
    "fathatan": "◌ً", "dammatan": "◌ٌ", "kasratan": "◌ٍ", "none": "◌",
}

RTL_CSS = """
<style>
[data-testid="stMain"] .stMarkdown, [data-testid="stMain"] label, [data-testid="stMain"] .stCaption,
[data-testid="stChatMessage"], [data-testid="stPopoverBody"] {direction: rtl; text-align: right;}
</style>
"""

PAGE_CSS = """
<style>
.arabic-big {font-size: 72px; direction: rtl; text-align: center; line-height: 1.6; margin: 0.2em 0;}
.arabic-mid {font-size: 44px; direction: rtl; text-align: center; line-height: 1.6;}
.tile {font-size: 52px; text-align: center; border: 2px solid #ccc; border-radius: 12px; line-height: 1.8;}
</style>
"""


# ---------- interface text ----------

# Returns interface text in the learner's language, filled with the given values.
def t(key: str, **values: Any) -> str:
    table = st.session_state.get("ui_text") or UI_TEXT["en"]
    return table.get(key, UI_TEXT["en"].get(key, key)).format(**values)


# Shows a vowel choice as its mark on a dotted circle plus its name in the learner's language.
def vowel_label(name: str) -> str:
    return f"{VOWEL_MARKS.get(name, '')} {t(name)}"


# ---------- graph ----------

# Wires every node and router into the tutor graph and compiles it with the checkpointer.
def build_graph(nodes: TutorNodes, checkpointer: BaseCheckpointSaver) -> Any:
    graph = StateGraph(TutorState)
    for name in ["load_session", "ask_field", "onboarding", "load_lesson", "teacher", "wait_for_learner",
                 "route_message", "offer_switch", "confirm_switch", "apply_switch", "decline_switch",
                 "offer_test_switch", "confirm_switch_mid_test", "build_test", "present_item", "read_letters",
                 "confirm_read", "retry_item", "skip_item", "check_letters", "ask_harakat", "grade_item",
                 "next_item", "finish_test", "test_chat", "exit_test"]:
        graph.add_node(name, getattr(nodes, name))
    test_steps = ["present_item", "confirm_read", "ask_harakat"]
    graph.add_edge(START, "load_session")
    graph.add_conditional_edges("load_session", nodes.route_after_session, ["ask_field", "load_lesson"])
    graph.add_edge("ask_field", "onboarding")
    graph.add_conditional_edges("onboarding", nodes.route_after_onboarding, ["ask_field", "load_lesson"])
    graph.add_conditional_edges("load_lesson", nodes.route_after_load, ["offer_switch", "teacher"])
    graph.add_edge("teacher", "wait_for_learner")
    graph.add_conditional_edges("wait_for_learner", nodes.route_after_wait, ["build_test", "route_message"])
    graph.add_conditional_edges("route_message", nodes.route_after_message,
                                ["teacher", "build_test", "offer_switch", "test_chat", "offer_test_switch", "exit_test"])
    graph.add_edge("offer_switch", "confirm_switch")
    graph.add_conditional_edges("confirm_switch", nodes.route_after_confirm_switch,
                                ["route_message", "confirm_switch", "apply_switch", "decline_switch"])
    graph.add_edge("apply_switch", "teacher")
    graph.add_edge("decline_switch", "wait_for_learner")
    graph.add_edge("offer_test_switch", "confirm_switch_mid_test")
    graph.add_conditional_edges("confirm_switch_mid_test", nodes.route_after_test_switch,
                                ["route_message", "exit_test", "confirm_switch_mid_test"] + test_steps)
    graph.add_edge("build_test", "present_item")
    graph.add_conditional_edges("present_item", nodes.route_after_present, ["read_letters", "present_item", "route_message"])
    graph.add_edge("read_letters", "confirm_read")
    graph.add_conditional_edges("confirm_read", nodes.route_after_confirm,
                                ["confirm_read", "check_letters", "retry_item", "skip_item", "route_message"])
    graph.add_edge("retry_item", "present_item")
    graph.add_edge("skip_item", "next_item")
    graph.add_conditional_edges("check_letters", nodes.route_after_check, ["ask_harakat", "grade_item"])
    graph.add_conditional_edges("ask_harakat", nodes.route_after_harakat, ["ask_harakat", "grade_item", "route_message"])
    graph.add_conditional_edges("test_chat", nodes.route_after_test_chat, test_steps)
    graph.add_conditional_edges("exit_test", nodes.route_after_exit, ["apply_switch", "offer_switch", "teacher"])
    graph.add_edge("grade_item", "next_item")
    graph.add_conditional_edges("next_item", nodes.route_after_next, ["present_item", "finish_test"])
    graph.add_edge("finish_test", "load_lesson")
    return graph.compile(checkpointer=checkpointer)


# Opens a Postgres-backed checkpointer, or a MemorySaver with a warning when no DB URL is set.
def make_checkpointer(db_url: str | None) -> tuple[BaseCheckpointSaver, str | None]:
    if not db_url:
        return MemorySaver(), "SUPABASE_DB_URL is not set: sessions are kept in memory and lost on restart."
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # Checks each connection before use and recycles idle ones, because Supabase closes idle connections.
    pool = ConnectionPool(
        db_url, min_size=1, max_size=5, max_idle=120, max_lifetime=1800, open=True,
        check=ConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row,
                "keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 3},
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver, None


# One shared checkpointer for the whole server process.
@st.cache_resource
def get_checkpointer() -> tuple[BaseCheckpointSaver, str | None]:
    return make_checkpointer(os.environ.get("SUPABASE_DB_URL"))


# One shared set of Cloudflare models for the whole server process.
@st.cache_resource
def get_models() -> Models:
    return Models()


# One Database per browser session, because it carries that user's auth session.
def get_db() -> Database:
    if "db" not in st.session_state:
        st.session_state.db = Database()
    return st.session_state.db


# Builds this user's nodes and graph once per browser session.
def get_graph() -> Any:
    if "graph" not in st.session_state:
        st.session_state.nodes = TutorNodes(get_db(), get_models())
        st.session_state.graph = build_graph(st.session_state.nodes, get_checkpointer()[0])
    return st.session_state.graph


# One thread per learner per day, so the session survives reloads.
def thread_id(user_id: str) -> str:
    return f"{user_id}:{datetime.date.today().isoformat()}"


# Run config with the thread id plus LangSmith metadata and tags.
def graph_config(user_id: str, values: dict[str, Any]) -> dict[str, Any]:
    lesson_id = (values.get("lesson") or {}).get("lesson_id") or "none"
    mode = values.get("mode") or "learn"
    return {
        "configurable": {"thread_id": thread_id(user_id)},
        "metadata": {"user_id": user_id, "lesson_id": lesson_id, "mode": mode},
        "tags": [f"user:{user_id}", f"lesson:{lesson_id}", f"mode:{mode}"],
        "run_name": "arabic-tutor",
    }


# Runs the graph with an input or resume command, keeping any error for display.
def run_graph(user_id: str, graph_input: Any) -> None:
    graph = get_graph()
    values = graph.get_state({"configurable": {"thread_id": thread_id(user_id)}}).values
    st.session_state.graph_error = None
    try:
        with st.spinner("…"):
            graph.invoke(graph_input, graph_config(user_id, values))
    except Exception as error:
        st.session_state.graph_error = str(error)


# Resumes the graph from its pending interrupt with a payload, then redraws the page.
def resume(user_id: str, payload: dict[str, Any]) -> None:
    run_graph(user_id, Command(resume=payload))
    st.rerun()


# Returns the value of the pending interrupt (what the graph is waiting for), or None.
def pending_interrupt(snapshot: Any) -> dict[str, Any] | None:
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


# ---------- account ----------

# Signs the user in, makes sure a profile row exists, and remembers the user.
def handle_sign_in(email: str, password: str) -> None:
    db = get_db()
    db.sign_in(email, password)
    user = db.get_user()
    if user and db.get_profile(user["id"]) is None:
        db.upsert_profile(user["id"], display_name=user.get("display_name"))
    st.session_state.user = user


# Draws the sign-in and sign-up forms for logged-out visitors.
def render_auth_forms() -> None:
    sign_in_tab, sign_up_tab = st.tabs([t("sign_in"), t("sign_up")])
    with sign_in_tab, st.form("sign_in"):
        email = st.text_input(t("email"), key="sign_in_email")
        password = st.text_input(t("password"), type="password", key="sign_in_password")
        if st.form_submit_button(t("sign_in")):
            try:
                handle_sign_in(email, password)
                st.rerun()
            except Exception as error:
                st.error(f"{t('sign_in_failed')}: {error}")
    with sign_up_tab, st.form("sign_up"):
        name = st.text_input(t("name"), key="sign_up_name")
        email = st.text_input(t("email"), key="sign_up_email")
        password = st.text_input(t("password"), type="password", key="sign_up_password")
        if st.form_submit_button(t("create_account")):
            try:
                user_id = get_db().sign_up(email, password, name)
                if user_id is None:
                    st.info(t("check_email"))
                else:
                    st.session_state.user = get_db().get_user()
                    st.rerun()
            except Exception as error:
                st.error(f"{t('sign_up_failed')}: {error}")


# Draws the logged-in account settings and saves changes to the profile.
def render_account_settings(user: dict[str, Any], waiting_kind: str | None) -> None:
    db = get_db()
    profile = db.get_profile(user["id"]) or {}
    name = st.text_input(t("name"), value=profile.get("display_name") or "")
    types = ["child", "non_native"]
    current_type = profile.get("learner_type")
    labels = [t(kind) for kind in types]
    chosen = st.selectbox(t("learner_type"), labels, index=types.index(current_type) if current_type in types else 0)
    learner_type = types[labels.index(chosen)]
    language = st.text_input(t("explain_language"), value=profile.get("explain_language") or "")
    reveal = st.slider(t("reveal_seconds"), 1, 30, int(profile.get("reveal_seconds") or 5))
    if st.button(t("save")):
        saved = db.upsert_profile(user["id"], display_name=name, learner_type=learner_type,
                                  explain_language=language.strip().lower() or None, reveal_seconds=reveal)
        sync_profile_into_graph(user["id"], saved, waiting_kind)
        st.rerun()
    if st.button(t("sign_out")):
        db.sign_out()
        st.session_state.clear()
        st.rerun()


# Copies saved profile changes into the graph state while the learner is in learn mode.
def sync_profile_into_graph(user_id: str, profile: dict[str, Any], waiting_kind: str | None) -> None:
    if waiting_kind != "learn":
        st.toast(t("saved_later"))
        return
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id(user_id)}}
    graph.update_state(config, {"profile": profile})


# Draws the Account popover in the top-left corner.
def render_account(waiting_kind: str | None) -> None:
    with st.popover(t("account")):
        user = st.session_state.get("user")
        if user is None:
            render_auth_forms()
        else:
            render_account_settings(user, waiting_kind)


# ---------- left column: cards ----------

# Draws big right-to-left Arabic text.
def arabic_text(text: str, css_class: str = "arabic-big") -> None:
    st.markdown(f'<div class="{css_class}">{html.escape(text)}</div>', unsafe_allow_html=True)


# Caches card glosses so each word is transliterated and translated once.
@st.cache_data(show_spinner=False)
def cached_gloss(text: str, meaning_en: str | None, language_code: str | None) -> str:
    return st.session_state.nodes.gloss_item(text, meaning_en, language_code)


# Learn mode card: one lesson item, with gloss for non-natives, and a Next button.
def render_word_card(lesson: dict[str, Any], profile: dict[str, Any]) -> None:
    items = lesson_items(lesson)
    if not items:
        return
    index = st.session_state.get("card_index", 0) % len(items)
    item = items[index]
    with st.container(border=True):
        st.caption(f"{lesson['lesson_id']} · {lesson['title_en']} · {lesson['title_ar']}")
        arabic_text(item["text"])
        if profile.get("learner_type") == "non_native":
            try:
                st.caption(cached_gloss(item["text"], item.get("meaning_en"), profile.get("explain_language")))
            except Exception as error:
                st.caption(t("gloss_unavailable"))
        if st.button(t("next"), key="next_card"):
            st.session_state.card_index = index + 1
            st.rerun()


# Starts a new reveal timer whenever a new item or attempt is shown.
def reset_reveal_if_new(info: dict[str, Any], reveal_seconds: int) -> None:
    reveal_key = f"{info['index']}-{info['attempt']}"
    if st.session_state.get("reveal_key") != reveal_key:
        st.session_state.reveal_key = reveal_key
        st.session_state.reveal_until = time.time() + reveal_seconds
        st.session_state.reveal_count = 1


# Test card: shows the item for reveal_seconds, then hides it; reruns every second.
@st.fragment(run_every=1)
def render_test_card(info: dict[str, Any], reveal_seconds: int) -> None:
    reset_reveal_if_new(info, reveal_seconds)
    remaining = st.session_state.reveal_until - time.time()
    with st.container(border=True):
        st.caption(t("item_of", k=info["index"] + 1, n=info["total"]))
        if remaining > 0:
            arabic_text(info["target"])
            st.progress(min(remaining / reveal_seconds, 1.0), text=f"{int(remaining) + 1}s")
        else:
            arabic_text("؟")
        if st.button(t("show_again"), key="show_again"):
            st.session_state.reveal_until = time.time() + reveal_seconds
            st.session_state.reveal_count += 1
            st.rerun(scope="fragment")


# Draws the item counter only, for the confirm and harakat steps.
def render_progress_card(info: dict[str, Any]) -> None:
    with st.container(border=True):
        st.caption(t("item_of", k=info["index"] + 1, n=info["total"]))


# ---------- left column: canvas ----------

# Draws a free-draw canvas; the Clear button gives it a new key so it resets.
def draw_canvas(base_key: str, disabled: bool = False) -> Any:
    from streamlit_drawable_canvas import st_canvas

    clear_count = st.session_state.get(f"clear-{base_key}", 0)
    stroke = st.session_state.get("stroke_width", 8)
    return st_canvas(
        drawing_mode="freedraw", stroke_color="#000000", background_color="#FFFFFF",
        stroke_width=stroke, height=280, width=560, key=f"{base_key}-{clear_count}",
        return_image_data=True, disabled=disabled,
    )


# Bumps the canvas key counter so the canvas is cleared on the next run.
def clear_canvas(base_key: str) -> None:
    st.session_state[f"clear-{base_key}"] = st.session_state.get(f"clear-{base_key}", 0) + 1


# Pen width slider shared by both canvases; defaults to 8.
def render_pen_slider() -> None:
    if "stroke_width" not in st.session_state:
        st.session_state.stroke_width = 8
    st.slider(t("pen_width"), 2, 30, key="stroke_width")


# Learn mode canvas: free practice, never graded.
def render_practice_canvas() -> None:
    render_pen_slider()
    draw_canvas("practice")
    st.caption(t("pen_hint"))
    if st.button(t("clear"), key="clear_practice"):
        clear_canvas("practice")
        st.rerun()


# Test canvas: Submit uploads PNG + strokes and resumes the graph with their paths.
def render_answer_canvas(user_id: str, info: dict[str, Any]) -> None:
    base_key = f"answer-{info['index']}-{info['attempt']}"
    render_pen_slider()
    result = draw_canvas(base_key)
    st.caption(t("pen_hint"))
    submit_col, clear_col = st.columns(2)
    if clear_col.button(t("clear"), key=f"clear_{base_key}"):
        clear_canvas(base_key)
        st.rerun()
    if submit_col.button(t("submit"), type="primary", key=f"submit_{base_key}"):
        if is_canvas_empty(result.json_data) or result.image_data is None:
            st.warning(t("empty_canvas"))
            return
        png_bytes = canvas_to_png(result.image_data)
        image_path, strokes_path = get_db().upload_attempt_files(user_id, png_bytes, result.json_data)
        st.session_state.last_png = png_bytes
        resume(user_id, {"type": "canvas", "image_path": image_path, "strokes_path": strokes_path,
                         "reveal_count": st.session_state.get("reveal_count", 1)})


# "I read" step: shows the OCR text with Yes / No buttons under the submitted image.
def render_confirm_read(user_id: str, info: dict[str, Any], is_child: bool) -> None:
    if st.session_state.get("last_png"):
        st.image(st.session_state.last_png, width=360)
    st.markdown(f"**{t('i_read')}**")
    arabic_text(info.get("ocr_text") or "—", "arabic-mid")
    yes_col, no_col = st.columns(2)
    if yes_col.button(t("yes_wrote"), type="primary"):
        resume(user_id, {"type": "confirm_read", "ok": True})
    if no_col.button(t("no_wrote")):
        resume(user_id, {"type": "confirm_read", "ok": False})


# Harakat tap input: one tile per letter (right to left) with mark choices; returns marks on Check.
def render_harakat_input(letters: list[str], allowed: dict[str, Any], key_prefix: str) -> list[dict[str, Any]] | None:
    columns = st.columns(len(letters))
    marks: list[dict[str, Any]] = [{} for _ in letters]
    for column_number, column in enumerate(columns):
        position = len(letters) - 1 - column_number
        with column:
            marks[position] = render_one_tile(letters[position], allowed, f"{key_prefix}-{position}")
    if st.button(t("check"), type="primary", key=f"check-{key_prefix}"):
        return marks
    return None


# Draws one letter tile with its vowel choices and optional shadda toggle.
def render_one_tile(letter: str, allowed: dict[str, Any], key: str) -> dict[str, Any]:
    shadda_key, vowel_key = f"{key}-shadda", f"{key}-vowel"
    shadda = bool(st.session_state.get(shadda_key, False))
    vowel = st.session_state.get(vowel_key) or "none"
    st.markdown(f'<div class="tile">{html.escape(letter_with_marks(letter, shadda, vowel))}</div>',
                unsafe_allow_html=True)
    if allowed.get("shadda"):
        st.toggle(t("shadda"), key=shadda_key)
    st.radio("mark", allowed["vowels"], index=None, key=vowel_key,
             format_func=vowel_label, label_visibility="collapsed")
    return {"shadda": bool(st.session_state.get(shadda_key, False)),
            "vowel": st.session_state.get(vowel_key) or "none"}


# Harakat step wrapper: draws the tiles and resumes the graph with the tapped marks.
def render_harakat_step(user_id: str, info: dict[str, Any]) -> None:
    st.markdown(f"**{t('tap_marks')}**")
    marks = render_harakat_input(info["letters"], info["allowed"], f"h-{info['index']}-{info['attempt']}")
    if marks is not None:
        resume(user_id, {"type": "harakat", "marks": marks})


# ---------- right column: chat ----------

# Draws the chat history in a fixed-height scrolling box.
def render_messages(messages: list[BaseMessage]) -> None:
    with st.container(height=520):
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            st.chat_message(role).markdown(str(message.content))


# Draws quick-reply buttons for onboarding options.
def render_onboarding_options(user_id: str, waiting: dict[str, Any]) -> None:
    for number, option in enumerate(waiting.get("options") or []):
        if st.button(option["label"], key=f"option-{waiting['field']}-{number}"):
            resume(user_id, {"type": "message", "text": option["text"]})


# Draws the chat panel: history, quick replies, chat input and the "I'm ready" button.
def render_chat(user_id: str, messages: list[BaseMessage], waiting: dict[str, Any] | None) -> None:
    render_messages(messages)
    kind = (waiting or {}).get("kind")
    if kind == "onboarding":
        render_onboarding_options(user_id, waiting)
    text = st.chat_input(t("write_message"), disabled=kind is None)
    if text:
        resume(user_id, {"type": "message", "text": text})
    if kind == "learn" and st.button(t("ready"), type="primary"):
        resume(user_id, {"type": "ready"})
    if kind == "confirm_switch":
        render_switch_buttons(user_id, waiting)
    if kind == "confirm_switch_mid_test":
        render_test_switch_buttons(user_id)


# Shows Yes / No for one offered lesson, or both lesson titles plus Neither for two.
def render_switch_buttons(user_id: str, waiting: dict[str, Any]) -> None:
    options = waiting.get("options") or []
    if len(options) == 1:
        labels = [(t("yes"), options[0]["lesson_id"]), (t("no"), None)]
    else:
        labels = [(option["title"], option["lesson_id"]) for option in options] + [(t("neither"), None)]
    for column, (label, lesson_id) in zip(st.columns(len(labels)), labels):
        if column.button(label, key=f"switch_{lesson_id}", width="stretch",
                         type="primary" if lesson_id else "secondary"):
            resume(user_id, {"type": "switch_answer", "lesson_id": lesson_id})


# Shows "Finish the test first" and "Stop and switch" during a test.
def render_test_switch_buttons(user_id: str) -> None:
    finish, stop = st.columns(2)
    if finish.button(t("finish_test_first"), width="stretch", type="primary"):
        resume(user_id, {"type": "test_switch_answer", "choice": "finish"})
    if stop.button(t("stop_and_switch"), width="stretch"):
        resume(user_id, {"type": "test_switch_answer", "choice": "stop"})


# ---------- page ----------

# Starts today's thread on first visit, or shows a retry button after an error.
def ensure_thread(user_id: str) -> Any:
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id(user_id)}}
    snapshot = graph.get_state(config)
    if not snapshot.values:
        run_graph(user_id, {"user_id": user_id, "messages": []})
        snapshot = graph.get_state(config)
    if st.session_state.get("graph_error"):
        st.error(f"{t('error')}: {st.session_state.graph_error}")
    if snapshot.next and pending_interrupt(snapshot) is None and st.button(t("retry")):
        run_graph(user_id, None)
        st.rerun()
    return snapshot


# Draws the left column for the current step: card on top, canvas below.
def render_left(user_id: str, values: dict[str, Any], waiting: dict[str, Any] | None) -> None:
    kind = (waiting or {}).get("kind")
    profile = values.get("profile") or {}
    if kind == "canvas":
        render_test_card(waiting, int(profile.get("reveal_seconds") or 5))
        render_answer_canvas(user_id, waiting)
    elif kind == "confirm_read":
        render_progress_card(waiting)
        render_confirm_read(user_id, waiting, profile.get("learner_type") == "child")
    elif kind == "harakat":
        render_progress_card(waiting)
        render_harakat_step(user_id, waiting)
    elif values.get("mode") == "test":
        st.caption(t("item_of", k=int(values.get("test_index") or 0) + 1, n=len(values.get("test_items") or [])))
    else:
        if values.get("lesson"):
            render_word_card(values["lesson"], profile)
        render_practice_canvas()


# Chooses the interface language from the profile (or the visitor's choice) and loads its texts.
def set_interface_language(profile: dict[str, Any] | None) -> str:
    if profile and profile.get("learner_type"):
        language = learner_language(profile)
    else:
        language = st.session_state.get("ui_lang", "ar")
    if language in UI_TEXT:
        st.session_state.ui_text = UI_TEXT[language]
    elif "nodes" in st.session_state:
        st.session_state.ui_text = st.session_state.nodes.translate_table(language, UI_TEXT["en"])
    else:
        st.session_state.ui_text = UI_TEXT["en"]
    if is_rtl(language):
        st.markdown(RTL_CSS, unsafe_allow_html=True)
    return language


# Lets a visitor who is not signed in pick Arabic or English for the interface.
def render_language_choice() -> None:
    choice = st.radio("العربية / English", ["ar", "en"], horizontal=True, key="ui_lang",
                      format_func=lambda code: "العربية" if code == "ar" else "English")
    st.session_state.ui_text = UI_TEXT[choice]


# Draws the whole page: Account on top, then the left column and the chat panel.
def main() -> None:
    st.set_page_config(page_title="Arabic spelling tutor · مُعَلِّمُ الإِمْلاءِ", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    warning = get_checkpointer()[1]
    user = st.session_state.get("user")
    if user is not None and not get_db().keep_session_fresh():
        st.session_state.pop("user", None)
        st.session_state.session_expired = True
        user = None
    if user is None:
        if st.session_state.pop("session_expired", False):
            st.warning(t("session_expired"))
        render_language_choice()
        set_interface_language(None)
        render_account(None)
        st.title(t("title"))
        st.info(t("open_account"))
        return
    get_graph()
    set_interface_language(get_db().get_profile(user["id"]))
    snapshot = ensure_thread(user["id"])
    set_interface_language(snapshot.values.get("profile"))
    waiting = pending_interrupt(snapshot)
    render_account((waiting or {}).get("kind"))
    if warning:
        st.warning(warning)
    left, right = st.columns([1, 1])
    with left:
        render_left(user["id"], snapshot.values, waiting)
    with right:
        render_chat(user["id"], snapshot.values.get("messages") or [], waiting)


if __name__ == "__main__":
    main()
