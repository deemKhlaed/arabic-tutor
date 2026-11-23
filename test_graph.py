"""Step-by-step tests of every graph node, router and helper, plus an end-to-end walk.

Run:  python test_graph.py            (fakes only)
      python test_graph.py --live     (also Supabase, OCR, chat, checkpointer, and the routing evaluation
                                       over data/routing_eval.jsonl with its three charts)
      python test_graph.py --no-plot  (skip the chart window)
"""

import argparse
import contextlib
import copy
import io
import json
import os
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from PIL import Image
from pydantic import Field

import Nodes
from Main import build_graph, make_checkpointer, pending_interrupt
from Nodes import TutorNodes, new_answer
from tools import (
    MARGIN,
    MIN_SIM,
    allowed_marks_for,
    build_card_text,
    is_fast_stay,
    parse_route,
    switch_options,
    build_system_prompt,
    build_test_items,
    canvas_to_png,
    compare_harakat,
    compare_letters,
    is_canvas_empty,
    lesson_items,
    normalize,
    split_letters_and_marks,
    strip_harakat,
)

load_dotenv()

# ---------- fixtures (copied from Supabase in their real shapes) ----------

LEVELS = [
    {"level": 0, "title_en": "Letters", "title_ar": "الحروف"},
    {"level": 1, "title_en": "All-fatha words", "title_ar": "كلمات بالفتحة"},
    {"level": 2, "title_en": "Words with harakat", "title_ar": "كلمات بالحركات"},
    {"level": 3, "title_en": "Sukun, shadda, madd, tanween, al, taa marbuta", "title_ar": "السكون والشدة والمد والتنوين وأل والتاء"},
    {"level": 4, "title_en": "Spelling rules", "title_ar": "قواعد الإملاء"},
]

L0_01 = {
    "lesson_id": "L0.01", "level": 0, "position": 1, "title_en": "Alif, baa, taa, thaa", "title_ar": "ا ب ت ث",
    "skill": "letters", "objective_en": "Recognize, name and write each letter in every position.",
    "writing_guide_en": "ب ت ث share one body; only the dots differ (1 below, 2 above, 3 above). Alif never joins the next letter.",
    "target_items": [
        {"letter": "ا", "name_ar": "ألف", "name_en": "alif", "sound_en": "long 'aa'", "forms": {"isolated": "ا", "initial": "ا", "medial": "ـا", "final": "ـا"}},
        {"letter": "ب", "name_ar": "باء", "name_en": "baa", "sound_en": "b", "forms": {"isolated": "ب", "initial": "بـ", "medial": "ـبـ", "final": "ـب"}},
        {"letter": "ت", "name_ar": "تاء", "name_en": "taa", "sound_en": "t", "forms": {"isolated": "ت", "initial": "تـ", "medial": "ـتـ", "final": "ـت"}},
        {"letter": "ث", "name_ar": "ثاء", "name_en": "thaa", "sound_en": "th", "forms": {"isolated": "ث", "initial": "ثـ", "medial": "ـثـ", "final": "ـث"}},
    ],
    "exercises": [{"type": "copy_letter", "items": ["ا", "ب", "ت", "ث"], "instruction_en": "Copy the letter you see."}],
    "pass_threshold": 0.85, "prerequisites": [], "rules": [],
}

L1_02 = {
    "lesson_id": "L1.02", "level": 1, "position": 2, "title_en": "All-fatha words, fully joined", "title_ar": "كلمات بالفتحة",
    "skill": "fatha_words", "objective_en": "Write three-letter words where every letter has fatha and all letters join.",
    "writing_guide_en": "Write the letters joined, right to left, and put a fatha above every letter: كَتَبَ = ka-ta-ba.",
    "target_items": [
        {"word": "فَتَحَ", "bare": "فتح", "letters": ["ف", "ت", "ح"], "meaning_en": "he opened"},
        {"word": "جَلَسَ", "bare": "جلس", "letters": ["ج", "ل", "س"], "meaning_en": "he sat"},
        {"word": "سَجَدَ", "bare": "سجد", "letters": ["س", "ج", "د"], "meaning_en": "he prostrated"},
        {"word": "كَتَبَ", "bare": "كتب", "letters": ["ك", "ت", "ب"], "meaning_en": "he wrote"},
        {"word": "طَبَخَ", "bare": "طبخ", "letters": ["ط", "ب", "خ"], "meaning_en": "he cooked"},
        {"word": "غَسَلَ", "bare": "غسل", "letters": ["غ", "س", "ل"], "meaning_en": "he washed"},
    ],
    "exercises": [{"type": "dictation_word", "items": ["فَتَحَ", "جَلَسَ", "سَجَدَ", "كَتَبَ", "طَبَخَ", "غَسَلَ"]}],
    "pass_threshold": 0.8, "prerequisites": ["L1.01"], "rules": [],
}

L4_03 = {
    "lesson_id": "L4.03", "level": 4, "position": 3, "title_en": "Middle hamza", "title_ar": "الهمزة المتوسطة",
    "skill": "book_03", "objective_en": "Choose the seat of a hamza in the middle of a word.",
    "writing_guide_en": None, "target_items": None,
    "exercises": [
        {"type": "explain_spelling", "items": ["قَائِل", "سَأَلَ"], "instruction_en": "Which two vowels did you compare?"},
        {"type": "dictation_word", "items": ["ذِئْب", "فُؤَاد", "مَسْأَلَة", "سُئِلَ", "شُؤُون"]},
    ],
    "pass_threshold": 0.8, "prerequisites": ["L4.02"],
    "rules": [
        {"rule_id": "R4.03.1", "lesson_id": "L4.03", "position": 1, "spelling": None, "examples": [], "non_examples": [],
         "condition_en": "Compare the vowel on the hamza with the vowel on the letter before it. The stronger one picks the seat."},
        {"rule_id": "R4.03.2", "lesson_id": "L4.03", "position": 2, "spelling": "ئ", "non_examples": [],
         "examples": ["بِئْر", "ذِئْب", "سُئِلَ", "قَائِل"],
         "condition_en": "Strongest vowel is kasra → written on a dotless yaa seat (ئ)."},
    ],
}

# Other target_items shapes, used only by the build_test_items tests.
SHAPE_L1_01 = {"level": 1, "target_items": [{"letter": "ب", "syllable": "بَ"}, {"letter": "ت", "syllable": "تَ"}]}
SHAPE_L2_01 = {"level": 2, "target_items": [{"letter": "ب", "fatha": "بَ", "kasra": "بِ", "damma": "بُ"}]}
SHAPE_L3_04 = {"level": 3, "target_items": [{"base": "بَاب", "meaning_en": "door",
                                             "forms": {"dammatan": "بَابٌ", "fathatan": "بَابًا", "kasratan": "بَابٍ"}}]}
SHAPE_L3_07 = {"level": 3, "target_items": {
    "taa_marbuta": [{"word": "شَجَرَةٌ", "meaning_en": "a tree"}],
    "taa_maftuha": [{"word": "بَيْتٌ", "meaning_en": "a house"}],
    "haa": [{"word": "وَجْهٌ", "meaning_en": "a face"}],
}}


# ---------- fakes ----------

class FakeDatabase:
    # In-memory stand-in for Database, seeded with three real lessons.
    def __init__(self) -> None:
        self.lessons = {lesson["lesson_id"]: copy.deepcopy(lesson) for lesson in (L0_01, L1_02, L4_03)}
        self.profiles: dict[str, dict[str, Any]] = {}
        self.progress: dict[tuple[str, str], dict[str, Any]] = {}
        self.attempts: list[dict[str, Any]] = []
        self.files: dict[str, bytes] = {}
        self.matches: list[dict[str, Any]] = []
        self.match_calls = 0

    # Pretends to sign up and returns a new user id.
    def sign_up(self, email: str, password: str, display_name: str) -> str:
        user_id = str(uuid.uuid4())
        self.upsert_profile(user_id, display_name=display_name)
        return user_id

    # Pretends to sign in and returns a fixed user id.
    def sign_in(self, email: str, password: str) -> str:
        return "user-1"

    # Does nothing, like a signed-out session.
    def sign_out(self) -> None:
        return None

    # The fake session never expires.
    def keep_session_fresh(self) -> bool:
        return True

    # Returns a fixed fake user.
    def get_user(self) -> dict[str, Any] | None:
        return {"id": "user-1", "email": "learner@example.com", "display_name": "Learner"}

    # Returns a copy of the stored profile, or None.
    def get_profile(self, user_id: str) -> dict[str, Any] | None:
        profile = self.profiles.get(user_id)
        return dict(profile) if profile else None

    # Merges fields into the stored profile and returns it.
    def upsert_profile(self, user_id: str, **fields: Any) -> dict[str, Any]:
        profile = self.profiles.setdefault(user_id, {"user_id": user_id, "reveal_seconds": 5})
        profile.update(fields)
        return dict(profile)

    # Returns the seeded levels.
    def get_levels(self) -> list[dict[str, Any]]:
        return copy.deepcopy(LEVELS)

    # Returns a copy of a seeded lesson with its rules.
    def get_lesson(self, lesson_id: str) -> dict[str, Any] | None:
        lesson = self.lessons.get(lesson_id)
        return copy.deepcopy(lesson) if lesson else None

    # Returns the first seeded lesson of a level.
    def get_first_lesson_of_level(self, level: int) -> dict[str, Any] | None:
        in_level = sorted((l for l in self.lessons.values() if l["level"] == level), key=lambda l: l["position"])
        return copy.deepcopy(in_level[0]) if in_level else None

    # Returns every seeded lesson with its rules, in course order.
    def get_course(self) -> list[dict[str, Any]]:
        return copy.deepcopy(sorted(self.lessons.values(), key=lambda l: (l["level"], l["position"])))

    # Returns the scripted retrieval result (the fake has no vectors).
    def match_lessons(self, embedding: list[float], k: int = 3) -> list[dict[str, Any]]:
        self.match_calls += 1
        return [dict(match) for match in self.matches[:k]]

    # Saves the current lesson in the stored profile.
    def set_current_lesson(self, user_id: str, lesson_id: str) -> None:
        self.upsert_profile(user_id, current_lesson_id=lesson_id)

    # Returns the next seeded lesson by (level, position), or None after the last.
    def get_next_lesson(self, lesson_id: str) -> dict[str, Any] | None:
        ordered = sorted(self.lessons.values(), key=lambda l: (l["level"], l["position"]))
        ids = [lesson["lesson_id"] for lesson in ordered]
        index = ids.index(lesson_id) + 1
        return copy.deepcopy(ordered[index]) if index < len(ordered) else None

    # Returns the learner's progress rows.
    def get_progress(self, user_id: str) -> list[dict[str, Any]]:
        return [dict(row) for (uid, _), row in self.progress.items() if uid == user_id]

    # Same rules as Database.save_test_result, in memory.
    def save_test_result(self, user_id: str, lesson_id: str, score: float, passed: bool) -> dict[str, Any]:
        previous = self.progress.get((user_id, lesson_id), {})
        best = max(score, previous.get("best_score") or 0.0)
        status = "passed" if passed or previous.get("status") == "passed" else "in_progress"
        row = {"user_id": user_id, "lesson_id": lesson_id, "status": status,
               "best_score": round(best, 3), "tests_taken": previous.get("tests_taken", 0) + 1}
        self.progress[(user_id, lesson_id)] = row
        return dict(row)

    # Stores one attempt row in memory.
    def record_attempt(self, **row: Any) -> dict[str, Any]:
        self.attempts.append(row)
        return row

    # Stores both files in memory and returns their paths.
    def upload_attempt_files(self, user_id: str, png_bytes: bytes, strokes_json: Any) -> tuple[str, str]:
        file_id = uuid.uuid4().hex
        self.files[f"{user_id}/{file_id}.png"] = png_bytes
        self.files[f"{user_id}/{file_id}.json"] = json.dumps(strokes_json).encode()
        return f"{user_id}/{file_id}.png", f"{user_id}/{file_id}.json"

    # Returns stored PNG bytes.
    def download_attempt_image(self, image_path: str) -> bytes:
        return self.files[image_path]


# Names the kind of request a prompt is, so the scripted chat can answer it.
def prompt_kind(messages: list[BaseMessage]) -> str:
    system = " ".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
    if "You route one message" in system:
        return "router"
    if "ISO 639-1" in system:
        return "language"
    if "examiner during a spelling test" in system:
        return "test_chat"
    if "Translate every value" in system:
        return "translate"
    if "examiner" in system:
        return "feedback"
    if "welcoming a new learner" in system:
        return "onboarding"
    if "transliteration, then its meaning" in system:
        return "gloss"
    return "teacher"


STAY = '{"intent": "stay", "lesson_id": null, "unsure": false}'
DEFAULT_REPLIES = {"router": STAY, "language": "XX", "feedback": "Good effort.",
                   "onboarding": "Which one are you?", "gloss": "kataba — he wrote", "translate": "{}",
                   "test_chat": "Write the word you saw on the canvas.", "teacher": "Let us learn the lesson."}


class ScriptedChat(BaseChatModel):
    replies: dict[str, list[str]] = Field(default_factory=dict)
    calls: list[tuple[str, list[BaseMessage]]] = Field(default_factory=list)

    # Name used by LangChain for this model type.
    @property
    def _llm_type(self) -> str:
        return "scripted"

    # Answers from the per-kind reply queue, or the default for that kind; records every call.
    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        kind = prompt_kind(messages)
        self.calls.append((kind, list(messages)))
        queue = self.replies.get(kind) or []
        text = queue.pop(0) if queue else DEFAULT_REPLIES[kind]
        if text.startswith("SLEEP:"):
            time.sleep(float(text.split(":")[1]))
            text = STAY
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    # Returns the system prompt of the most recent call of a kind.
    def last_system(self, kind: str) -> str:
        for call_kind, messages in reversed(self.calls):
            if call_kind == kind:
                return " ".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
        return ""

    # Returns the human text of the latest call of a given kind.
    def last_system_input(self, kind: str) -> str:
        for call_kind, messages in reversed(self.calls):
            if call_kind == kind:
                return " ".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
        return ""


class FakeModels:
    # Scripted chat plus an OCR that returns preset strings in order.
    def __init__(self) -> None:
        self.chat = ScriptedChat()
        self.ocr_outputs: list[str] = []
        self.embedded: list[str] = []

    # Records the texts and returns tiny fake vectors (retrieval itself is scripted in FakeDatabase).
    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[0.0, 0.0, 0.0] for _ in texts]

    # Returns the next preset OCR string with harakat stripped, like the real Models.
    def read_handwriting(self, png_bytes: bytes) -> str:
        return strip_harakat(self.ocr_outputs.pop(0))


# Patches Nodes.interrupt so an interrupt node receives the given resume payload.
@contextlib.contextmanager
def resume_with(payload: Any) -> Iterator[list[Any]]:
    seen: list[Any] = []
    original = Nodes.interrupt
    Nodes.interrupt = lambda value: (seen.append(value), payload)[1]
    try:
        yield seen
    finally:
        Nodes.interrupt = original


# Builds fresh fakes and nodes for a unit test.
def make_nodes(test_size: int = 5, seed: int | None = 1) -> tuple[TutorNodes, FakeDatabase, FakeModels]:
    db, models = FakeDatabase(), FakeModels()
    return TutorNodes(db, models, test_size=test_size, seed=seed), db, models


# A test state in the middle of a test on the given lesson.
def test_state(lesson: dict[str, Any], items: list[str], answer: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    state = {"user_id": "user-1", "profile": {"learner_type": "child", "explain_language": "ar"},
             "lesson": copy.deepcopy(lesson), "mode": "test", "test_items": items, "test_index": 0,
             "current_answer": answer or new_answer(), "test_results": [], "messages": []}
    state.update(extra)
    return state


# Raises AssertionError with a message when a condition is false.
def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ---------- unit tests: session and onboarding ----------

# New user: empty profile, all three fields missing.
def t_load_session_new() -> str:
    nodes, _, _ = make_nodes()
    out = nodes.load_session({"user_id": "user-1"})
    check(out["missing_fields"] == ["learner_type", "start_level"], f"missing={out['missing_fields']}")
    return "missing = learner_type, start_level"


# Returning child without explain_language gets 'ar' and nothing missing.
def t_load_session_returning() -> str:
    nodes, db, _ = make_nodes()
    db.upsert_profile("user-1", learner_type="child", start_level=0)
    out = nodes.load_session({"user_id": "user-1"})
    check(out["missing_fields"] == [], f"missing={out['missing_fields']}")
    check(out["profile"]["explain_language"] == "ar", "child should default to ar")
    return "complete profile, explain_language=ar"


# Asking a field appends one AI question and offers the level list.
def t_ask_field() -> str:
    nodes, _, models = make_nodes()
    out = nodes.ask_field({"user_id": "user-1", "missing_fields": ["start_level"], "profile": {"learner_type": "child"}})
    check(isinstance(out["messages"][0], AIMessage), "expected an AI question")
    check("Spelling rules" in models.chat.last_system("onboarding"), "levels missing from prompt")
    return "question asked, levels offered"


# Answering "child" saves learner_type and explain_language=ar.
def t_onboarding_child() -> str:
    nodes, db, _ = make_nodes()
    state = {"user_id": "user-1", "missing_fields": ["learner_type", "start_level"], "profile": {}}
    with resume_with({"type": "message", "text": "child"}):
        out = nodes.onboarding(state)
    check(db.profiles["user-1"]["learner_type"] == "child", "learner_type not saved")
    check(db.profiles["user-1"]["explain_language"] == "ar", "explain_language not defaulted")
    check(out["missing_fields"] == ["start_level"], f"missing={out['missing_fields']}")
    return "child → ar, next: start_level"


# Non-native then a language name; also the LLM fallback for an unlisted language name.
def t_onboarding_language() -> str:
    nodes, db, models = make_nodes()
    with resume_with({"type": "message", "text": "I am a non-native speaker"}):
        out = nodes.onboarding({"user_id": "user-1", "missing_fields": ["learner_type", "start_level"], "profile": {}})
    check(out["missing_fields"] == ["explain_language", "start_level"], f"missing={out['missing_fields']}")
    with resume_with({"type": "message", "text": "Urdu please"}):
        nodes.onboarding({"user_id": "user-1", "missing_fields": out["missing_fields"], "profile": out["profile"]})
    check(db.profiles["user-1"]["explain_language"] == "ur", "Urdu not parsed")
    models.chat.replies["language"] = ["sv"]
    with resume_with({"type": "message", "text": "svenska"}):
        nodes.onboarding({"user_id": "user-1", "missing_fields": ["explain_language", "start_level"], "profile": {}})
    check(db.profiles["user-1"]["explain_language"] == "sv", "LLM fallback code not saved")
    return "non_native → ur; svenska → sv via LLM"


# "I don't know" means level 0 and onboarding is complete.
def t_onboarding_level() -> str:
    nodes, db, _ = make_nodes()
    db.upsert_profile("user-1", learner_type="child", explain_language="ar")
    with resume_with({"type": "message", "text": "I don't know"}) as seen:
        out = nodes.onboarding({"user_id": "user-1", "missing_fields": ["start_level"], "profile": {}})
    check(db.profiles["user-1"]["start_level"] == 0, "start_level should be 0")
    check(out["missing_fields"] == [], "nothing should be missing")
    check(len(seen[0]["options"]) == 6, "five levels + I don't know expected")
    check(nodes.route_after_onboarding(out) == "load_lesson", "should go to load_lesson")
    return "I don't know → 0 → load_lesson"


# An invalid answer saves nothing and sets a note for the next question.
def t_onboarding_invalid() -> str:
    nodes, db, _ = make_nodes()
    with resume_with({"type": "message", "text": "banana"}):
        out = nodes.onboarding({"user_id": "user-1", "missing_fields": ["start_level"], "profile": {}})
    check("user-1" not in db.profiles, "nothing should be saved")
    check("banana" in out["onboarding_note"], "note should mention the answer")
    check(nodes.route_after_onboarding({"missing_fields": ["start_level"]}) == "ask_field", "should ask again")
    return "rejected, asks again"


# ---------- unit tests: learning ----------

# Loads the first lesson of the start level and stores it as current.
def t_load_lesson() -> str:
    nodes, db, _ = make_nodes()
    db.upsert_profile("user-1", learner_type="child", start_level=1)
    out = nodes.load_lesson({"user_id": "user-1", "profile": db.get_profile("user-1")})
    check(out["lesson"]["lesson_id"] == "L1.02", f"got {out['lesson']['lesson_id']}")
    check(db.profiles["user-1"]["current_lesson_id"] == "L1.02", "current_lesson_id not saved")
    check(out["mode"] == "learn", "mode should be learn")
    out_again = nodes.load_lesson({"user_id": "user-1", "profile": db.get_profile("user-1")})
    check(out_again["lesson"]["lesson_id"] == "L1.02", "should reuse current_lesson_id")
    return "start_level 1 → L1.02"


# Child and non-native prompts differ; a reply without a question gets the readiness question.
def t_teacher() -> str:
    nodes, _, models = make_nodes()
    child = {"learner_type": "child", "explain_language": "ar"}
    adult = {"learner_type": "non_native", "explain_language": "ur"}
    out_child = nodes.teacher({"profile": child, "lesson": L1_02, "messages": []})
    child_prompt = models.chat.last_system("teacher")
    out_adult = nodes.teacher({"profile": adult, "lesson": L1_02, "messages": [HumanMessage("hi")]})
    adult_prompt = models.chat.last_system("teacher")
    check(child_prompt != adult_prompt, "prompts should differ")
    check("Arabic-speaking child" in child_prompt and "60 words" in child_prompt, "child style missing")
    check("Urdu" in adult_prompt and "transliteration" in adult_prompt, "non-native style missing")
    check(str(out_child["messages"][0].content).endswith("؟"), "child reply must end with Arabic question")
    check(str(out_adult["messages"][0].content).endswith("?"), "adult reply must end with a question")
    models.chat.replies["teacher"] = ["Ready for the test?"]
    out = nodes.teacher({"profile": adult, "lesson": L1_02, "messages": [HumanMessage("ok")]})
    check(out["messages"][0].content == "Ready for the test?", "existing question should be kept")
    return "styles differ; every reply ends with a question"


# Builds a scripted retrieval result from (lesson_id, similarity) pairs.
def matches(*pairs: tuple[str, float]) -> list[dict[str, Any]]:
    return [{"lesson_id": lesson_id, "similarity": similarity} for lesson_id, similarity in pairs]


# Returns the JSON a router would answer.
def router_json(intent: str, lesson_id: str | None = None, unsure: bool = False) -> str:
    return json.dumps({"intent": intent, "lesson_id": lesson_id, "unsure": unsure})


# A minimal learn-mode state on L1.02 with one learner message, for calling route_message directly.
def message_state(text: str, mode: str = "learn") -> dict[str, Any]:
    return {"user_id": "user-1", "profile": {"learner_type": "non_native", "explain_language": "en"},
            "lesson": copy.deepcopy(L1_02), "mode": mode, "messages": [HumanMessage(text)]}


# Fast path: the current lesson leads clearly → stay with no router call; otherwise the router decides.
def t_fast_path() -> str:
    nodes, db, models = make_nodes()
    db.matches = matches(("L1.02", MIN_SIM + 0.2), ("L0.01", MIN_SIM))
    decision = nodes.route_message(message_state("why is kataba joined?"))["route"]
    check(decision["intent"] == "stay" and decision["fast_path"], f"fast path expected: {decision}")
    check(not models.chat.calls and db.match_calls == 1, "fast path must not call the router")
    check(models.embedded == ["why is kataba joined?"], "the message itself must be embedded")
    db.matches = matches(("L1.02", MIN_SIM - 0.01), ("L0.01", 0.1))
    check(not nodes.route_message(message_state("hmm"))["route"]["fast_path"], "below MIN_SIM → router")
    db.matches = matches(("L1.02", 0.8), ("L0.01", 0.8 - MARGIN / 2))
    check(not nodes.route_message(message_state("hmm"))["route"]["fast_path"], "lead below MARGIN → router")
    db.matches = matches(("L0.01", 0.9), ("L1.02", 0.5))
    check(not nodes.route_message(message_state("hmm"))["route"]["fast_path"], "current not on top → router")
    check(len(models.chat.calls) == 3 and all(kind == "router" for kind, _ in models.chat.calls), "3 router calls")
    check(is_fast_stay(matches(("L1.02", 0.6)), "L1.02"), "a single strong match is a fast stay")
    return f"MIN_SIM {MIN_SIM}, MARGIN {MARGIN}: 1 fast stay, 3 router calls"


# Router JSON parsing: valid answers, garbage, unknown ids, timeouts and the NEXT candidate.
def t_parse_route() -> str:
    ids = ["L4.03", "L0.01", "L1.02"]
    cases = [
        (router_json("switch", "L4.03"), ("switch", "L4.03", False)),
        ("```json\n" + router_json("ready") + "\n```", ("ready", None, False)),
        (router_json("chat", "L4.03"), ("chat", None, False)),
        (router_json("switch", None, True), ("switch", None, True)),
        ("not json at all", ("stay", None, False)),
        ('{"intent": "dance"}', ("stay", None, False)),
        (router_json("switch", "L9.99"), ("stay", None, False)),
        (router_json("switch", None, False), ("stay", None, False)),
        ('{"intent": "SWITCH", "lesson_id": "L0.01", "unsure": "true"}', ("switch", "L0.01", True)),
    ]
    for text, expected in cases:
        route = parse_route(text, ids)
        check((route["intent"], route["lesson_id"], route["unsure"]) == expected, f"{text!r} → {route}")
    nodes, db, models = make_nodes()
    original = Nodes.ROUTER_TIMEOUT_SECONDS
    Nodes.ROUTER_TIMEOUT_SECONDS = 0.2
    try:
        models.chat.replies["router"] = ["SLEEP:0.6"]
        decision = nodes.route_message(message_state("teach me hamza"))["route"]
    finally:
        Nodes.ROUTER_TIMEOUT_SECONDS = original
    check(decision["intent"] == "stay" and decision.get("error") == "TimeoutError", f"timeout → stay: {decision}")
    db.matches = matches(("L4.03", 0.7))
    models.chat.replies["router"] = [router_json("switch", "L0.01")]
    decision = nodes.route_message(message_state("teach me letters"))["route"]
    check(decision["intent"] == "stay", "a lesson that was not a candidate must become stay")
    db.matches = []
    models.chat.replies["router"] = [router_json("switch", "L4.03")]
    decision = nodes.route_message(message_state("something harder"))["route"]
    check(decision["intent"] == "switch" and decision["options"] == ["L4.03"], f"NEXT candidate: {decision}")
    check("[NEXT]" in models.chat.last_system_input("router"), "the next lesson must be offered as NEXT")
    return f"{len(cases)} parses, timeout → stay, non-candidate → stay, NEXT offered"


# switch_options: one lesson normally; two when unsure or when retrieval is too close to call.
def t_switch_options() -> str:
    route = {"intent": "switch", "lesson_id": "L4.03", "unsure": False}
    clear = matches(("L4.03", 0.7), ("L0.01", 0.5), ("L1.02", 0.4))
    close = matches(("L4.03", 0.7), ("L0.01", 0.7 - MARGIN / 2), ("L1.02", 0.4))
    check(switch_options(route, clear, "L1.02") == ["L4.03"], "clear → one option")
    check(switch_options(route, close, "L1.02") == ["L4.03", "L0.01"], "close → two options")
    check(switch_options({**route, "unsure": True}, clear, "L1.02") == ["L4.03", "L0.01"], "unsure → two options")
    check(switch_options({**route, "lesson_id": None, "unsure": True}, clear, "L1.02") == ["L4.03", "L0.01"], "unsure, no id")
    check(switch_options({**route, "lesson_id": "L1.02"}, clear, "L1.02") == [], "switch to the current lesson → none")
    return "1 / 2 options, current lesson excluded"


# Every route in the table: learn and test mode × stay / chat / ready / switch / stop.
def t_route_table() -> str:
    nodes, _, _ = make_nodes()
    expected = {("learn", "stay"): "teacher", ("learn", "chat"): "teacher", ("learn", "ready"): "build_test",
                ("learn", "switch"): "offer_switch", ("learn", "stop"): "teacher",
                ("test", "stay"): "test_chat", ("test", "chat"): "test_chat", ("test", "ready"): "test_chat",
                ("test", "switch"): "offer_test_switch", ("test", "stop"): "exit_test"}
    for (mode, intent), target in expected.items():
        got = nodes.route_after_message({"mode": mode, "route": {"intent": intent}})
        check(got == target, f"{mode}/{intent} → {got}, expected {target}")
    return f"{len(expected)} routes"


# Card text: titles, objective, skill, letters, marks, examples, rules (level 4) and the sample questions.
def t_card_text() -> str:
    lesson = copy.deepcopy(L4_03)
    lesson["rules"][0]["condition_ar"] = "قارن حركة الهمزة بحركة ما قبلها"
    card = build_card_text(lesson, ["متى اكتب الهمزه على واو"])
    for part in [lesson["title_ar"], lesson["title_en"], lesson["objective_en"], "skill:", "letters:",
                 "examples:", "قارن حركة الهمزة بحركة ما قبلها", "متى اكتب الهمزه على واو"]:
        check(part in card, f"card text misses {part!r}")
    check("marks:" in build_card_text(L1_02, []), "level 1 card should list marks")
    return f"{len(card)} characters for L4.03"


# ---------- unit tests: test building and text checks ----------

# build_test_items handles every target_items shape.
def t_build_test_items() -> str:
    expectations = {
        "L0 letters": (L0_01, {"ا", "ب", "ت", "ث"}),
        "L1.01 syllables": (SHAPE_L1_01, {"بَ", "تَ"}),
        "L2.01 harakat set": (SHAPE_L2_01, {"بَ", "بِ", "بُ"}),
        "word lesson": (L1_02, {item["word"] for item in L1_02["target_items"]}),
        "L3.04 tanween forms": (SHAPE_L3_04, {"بَابٌ", "بَابًا", "بَابٍ"}),
        "L3.07 grouped": (SHAPE_L3_07, {"شَجَرَةٌ", "بَيْتٌ", "وَجْهٌ"}),
        "L4 rules + dictation": (L4_03, {"بِئْر", "ذِئْب", "سُئِلَ", "قَائِل", "فُؤَاد", "مَسْأَلَة", "شُؤُون"}),
    }
    for name, (lesson, expected) in expectations.items():
        texts = {item["text"] for item in lesson_items(lesson)}
        check(texts == expected, f"{name}: {texts}")
        sample = build_test_items(lesson, n=5, seed=3)
        check(len(sample) == min(5, len(expected)) and len(set(sample)) == len(sample), f"{name}: bad sample {sample}")
        check(sample == build_test_items(lesson, n=5, seed=3), f"{name}: seed not deterministic")
    return f"{len(expectations)} shapes"


# normalize / strip_harakat / compare_letters cases from the spec.
def t_compare_letters() -> str:
    check(compare_letters("كَتَبَ", "كتب")["letters_correct"], "correct case failed")
    wrong = compare_letters("كَتَبَ", "كثب")
    check(wrong["errors"] == [{"kind": "letter", "position": 1, "expected": "ت", "got": "ث"}], f"wrong letter: {wrong}")
    taa = compare_letters("مَدْرَسَةٌ", "مدرسه")
    check(taa["errors"] == [{"kind": "letter", "position": 4, "expected": "ة", "got": "ه"}], f"ة vs ه: {taa}")
    yaa = compare_letters("عَلَى", "علي")
    check(not yaa["letters_correct"] and yaa["errors"][0]["expected"] == "ى", f"ى vs ي: {yaa}")
    check(compare_letters("كَتَبَ", "كَتَبَ")["letters_correct"], "OCR harakat must be stripped")
    check(normalize("كـتـب، ") == "كتب", "tatweel/space/punctuation not removed")
    check(compare_letters("أَكَلَ", "اكل")["letters_correct"] is False, "أ and ا must stay distinct")
    check(strip_harakat("بَابًا") == "بابا", "strip_harakat failed")
    return "5 cases + normalize"


# split_letters_and_marks / compare_harakat cases from the spec.
def t_compare_harakat() -> str:
    # Returns the correct taps for a word, as the UI would send them.
    def marks(text: str) -> list[dict[str, Any]]:
        return [{"shadda": e["shadda"], "vowel": e["vowel"]} for e in split_letters_and_marks(text)]

    check([e["vowel"] for e in split_letters_and_marks("كَتَبَ")] == ["fatha"] * 3, "all fatha")
    check([e["vowel"] for e in split_letters_and_marks("كُتِبَ")] == ["damma", "kasra", "fatha"], "kasra and damma")
    check([e["vowel"] for e in split_letters_and_marks("بَحْرٌ")] == ["fatha", "sukun", "dammatan"], "sukun")
    allam = split_letters_and_marks("عَلَّمَ")
    check(allam[1] == {"letter": "ل", "shadda": True, "vowel": "fatha"}, f"shadda+fatha: {allam}")
    check([e["vowel"] for e in split_letters_and_marks("بَابًا")] == ["fatha", "none", "fathatan", "none"], "tanween + alif")
    sun = split_letters_and_marks("الشَّمْسُ")
    check([(e["shadda"], e["vowel"]) for e in sun] == [(False, "none"), (False, "none"), (True, "fatha"), (False, "sukun"), (False, "damma")], f"sun ال: {sun}")
    for word in ["كَتَبَ", "كُتِبَ", "بَحْرٌ", "عَلَّمَ", "بَابًا", "الشَّمْسُ"]:
        check(compare_harakat(word, marks(word))["harakat_correct"], f"self-compare failed for {word}")
    tapped = marks("كَتَبَ")
    tapped[2]["vowel"] = "kasra"
    result = compare_harakat("كَتَبَ", tapped)
    check(result["errors"] == [{"kind": "haraka", "position": 2, "letter": "ب", "expected": "fatha", "got": "kasra"}], f"wrong mark: {result}")
    return "7 cases"


# Mark buttons offered per level.
def t_allowed_marks() -> str:
    check(allowed_marks_for(L1_02) == {"vowels": ["fatha", "none"], "shadda": False}, f"L1: {allowed_marks_for(L1_02)}")
    check(allowed_marks_for(SHAPE_L2_01)["vowels"] == ["fatha", "kasra", "damma", "none"], "L2")
    level3 = allowed_marks_for(SHAPE_L3_04)
    check(len(level3["vowels"]) == 8 and level3["shadda"], f"L3: {level3}")
    return "L1 fatha · L2 3 vowels · L3 all"


# canvas_to_png turns transparency white and crops to the ink; empty canvas detection.
def t_canvas_png() -> str:
    pixels = np.zeros((280, 500, 4), dtype=np.uint8)
    pixels[100:120, 200:260] = [0, 0, 0, 255]
    image = Image.open(io.BytesIO(canvas_to_png(pixels)))
    check(image.mode == "RGB", "PNG should be RGB")
    check(image.height == 256 and abs(image.width / image.height - (60 + 48) / (20 + 48)) < 0.02, f"crop size {image.size}")
    check(image.getpixel((2, 2)) == (255, 255, 255), "transparent must become white")
    check(is_canvas_empty({"objects": []}) and is_canvas_empty(None), "empty canvas not detected")
    check(not is_canvas_empty({"objects": [{"type": "path"}]}), "drawn canvas seen as empty")
    return f"cropped to {image.size}"


# Teacher system prompt lists level 4 rules and lesson items.
def t_system_prompt_rules() -> str:
    prompt = build_system_prompt({"learner_type": "child"}, L4_03)
    check("R4.03.2" in prompt and "ذِئْب" in prompt, "rules or items missing")
    check("ready" in prompt.lower(), "readiness instruction missing")
    return "rules + readiness present"


# ---------- unit tests: test flow ----------

# read_letters runs the OCR on the stored PNG and asks for confirmation; yes → check_letters.
def t_read_confirm_yes() -> str:
    nodes, db, models = make_nodes()
    image_path, strokes_path = db.upload_attempt_files("user-1", b"png", {"objects": [1]})
    answer = {**new_answer(), "image_path": image_path, "strokes_path": strokes_path}
    state = test_state(L1_02, ["كَتَبَ"], answer)
    models.ocr_outputs = ["كتب"]
    state["current_answer"] = nodes.read_letters(state)["current_answer"]
    check(state["current_answer"]["ocr_text"] == "كتب", "OCR text not stored")
    with resume_with({"type": "confirm_read", "ok": True}):
        state["current_answer"] = nodes.confirm_read(state)["current_answer"]
    check(nodes.route_after_confirm(state) == "check_letters", "yes → check_letters")
    return "OCR read, yes → check_letters"


# One "no": same item again on a fresh canvas, logged as uncounted.
def t_confirm_no_once() -> str:
    nodes, db, _ = make_nodes()
    state = test_state(L1_02, ["كَتَبَ"], {**new_answer(), "ocr_text": "كنب", "image_path": "p.png"})
    with resume_with({"type": "confirm_read", "ok": False}):
        state["current_answer"] = nodes.confirm_read(state)["current_answer"]
    check(nodes.route_after_confirm(state) == "retry_item", "first no → retry_item")
    out = nodes.retry_item(state)
    check(out["current_answer"]["attempt"] == 1 and out["current_answer"]["dispute_count"] == 1, "attempt/dispute not kept")
    check(db.attempts[-1]["counted"] is False and db.attempts[-1]["ocr_disputed"], "disputed read not logged")
    check(state["test_index"] == 0, "must stay on the same item")
    return "retry same item, attempt 1"


# Two "no"s: skipped, counted=false, then next item.
def t_confirm_no_twice() -> str:
    nodes, db, _ = make_nodes()
    answer = {**new_answer(attempt=1, dispute_count=1), "ocr_text": "كنب", "image_path": "p2.png"}
    state = test_state(L1_02, ["كَتَبَ"], answer)
    with resume_with({"type": "confirm_read", "ok": False}):
        state["current_answer"] = nodes.confirm_read(state)["current_answer"]
    check(nodes.route_after_confirm(state) == "skip_item", "second no → skip_item")
    out = nodes.skip_item(state)
    check(out["test_results"][0]["counted"] is False, "skipped item must not count")
    check(db.attempts[-1]["counted"] is False and db.attempts[-1]["image_path"] == "p2.png", "skip not logged with image")
    return "skipped, counted=false"


# check_letters routing: wrong letters → grade; level 2 right → harakat; level 4 → grade.
def t_check_letters_routing() -> str:
    nodes, _, _ = make_nodes()
    wrong = test_state(L1_02, ["كَتَبَ"], {**new_answer(), "ocr_text": "كثب"})
    wrong["current_answer"] = nodes.check_letters(wrong)["current_answer"]
    check(nodes.route_after_check(wrong) == "grade_item", "wrong letters → grade_item")
    level2 = test_state({**L1_02, "level": 2}, ["كُتِبَ"], {**new_answer(), "ocr_text": "كتب"})
    level2["current_answer"] = nodes.check_letters(level2)["current_answer"]
    check(nodes.route_after_check(level2) == "ask_harakat", "level 2 right → ask_harakat")
    level4 = test_state(L4_03, ["ذِئْب"], {**new_answer(), "ocr_text": "ذئب"})
    level4["current_answer"] = nodes.check_letters(level4)["current_answer"]
    check(nodes.route_after_check(level4) == "grade_item", "level 4 → grade_item")
    return "3 routes"


# ask_harakat offers per-letter tiles; grade_item combines results and records the attempt.
def t_grade_item() -> str:
    nodes, db, models = make_nodes()
    state = test_state(L1_02, ["كَتَبَ"], {**new_answer(), "ocr_text": "كتب", "letters_correct": True, "errors": []})
    taps = [{"shadda": False, "vowel": "fatha"}, {"shadda": False, "vowel": "kasra"}, {"shadda": False, "vowel": "fatha"}]
    with resume_with({"type": "harakat", "marks": taps}) as seen:
        state["current_answer"] = nodes.ask_harakat(state)["current_answer"]
    check(seen[0]["letters"] == ["ك", "ت", "ب"], f"tiles: {seen[0]['letters']}")
    out = nodes.grade_item(state)
    result = out["test_results"][0]
    check(result["letters_correct"] and result["harakat_correct"] is False and result["is_correct"] is False, f"result {result}")
    check(db.attempts[-1]["harakat_input"] == taps and db.attempts[-1]["counted"], "attempt not recorded")
    check("examiner" in models.chat.last_system("feedback"), "feedback prompt not used")
    level4 = test_state(L4_03, ["ذِئْب"], {**new_answer(), "ocr_text": "ذءب", "letters_correct": False,
                                         "errors": [{"kind": "letter", "position": 1, "expected": "ئ", "got": "ء"}]})
    out4 = nodes.grade_item(level4)
    check(out4["test_results"][0]["harakat_correct"] is None, "level 4 must not test harakat")
    check("R4.03.2" in models.chat.last_system("feedback"), "level 4 feedback must cite rules")
    return "graded, recorded, rules cited on L4"


# finish_test: pass → next lesson; fail → same lesson + review; all skipped → no score.
def t_finish_test() -> str:
    nodes, db, _ = make_nodes()
    db.upsert_profile("user-1", learner_type="child", current_lesson_id="L1.02")
    ok = {"counted": True, "is_correct": True, "target": "كَتَبَ"}
    bad = {"counted": True, "is_correct": False, "target": "جَلَسَ"}
    skipped = {"counted": False, "is_correct": None, "target": "فَتَحَ"}
    passed = nodes.finish_test(test_state(L1_02, [], test_results=[ok, ok, ok, ok, skipped]))
    check(passed["profile"]["current_lesson_id"] == "L4.03", "pass should move to next lesson")
    check(db.progress[("user-1", "L1.02")]["status"] == "passed", "progress not passed")
    db.upsert_profile("user-1", current_lesson_id="L1.02")
    failed = nodes.finish_test(test_state(L1_02, [], test_results=[ok, bad, bad]))
    check("profile" not in failed and failed["review_items"] == ["جَلَسَ", "جَلَسَ"], f"fail: {failed}")
    check(db.progress[("user-1", "L1.02")]["tests_taken"] == 2, "tests_taken not bumped")
    check(db.progress[("user-1", "L1.02")]["best_score"] == 1.0, "best score not kept")
    before = dict(db.progress)
    none = nodes.finish_test(test_state(L1_02, [], test_results=[skipped, skipped]))
    check(db.progress == before and none["mode"] == "learn", "all skipped must not save a score")
    return "pass → L4.03, fail stays, all skipped → no score"


# Starts a graph for a non-native learner on L1.02 and returns it, waiting in learn mode.
def learn_graph(thread: str, test_size: int = 2) -> tuple[Any, dict[str, Any], FakeDatabase, FakeModels]:
    db, models = FakeDatabase(), FakeModels()
    graph = build_graph(TutorNodes(db, models, test_size=test_size, seed=1), MemorySaver())
    config = {"configurable": {"thread_id": f"user-1:{thread}"}}
    db.upsert_profile("user-1", learner_type="non_native", explain_language="en", start_level=1)
    graph.invoke({"user_id": "user-1", "messages": []}, config)
    return graph, config, db, models


# Answers the current test item correctly: canvas, confirm, and (level 1) the right harakat.
def answer_item(graph: Any, config: dict[str, Any], db: FakeDatabase, models: FakeModels) -> tuple[Any, Any]:
    snapshot = graph.get_state(config)
    target = snapshot.values["test_items"][snapshot.values["test_index"]]
    models.ocr_outputs = [target]
    step(graph, config, canvas_payload(db))
    snapshot, waiting = step(graph, config, {"type": "confirm_read", "ok": True})
    if waiting and waiting["kind"] == "harakat":
        marks = [{"shadda": e["shadda"], "vowel": e["vowel"]} for e in split_letters_and_marks(target)]
        snapshot, waiting = step(graph, config, {"type": "harakat", "marks": marks})
    return snapshot, waiting


# Learn mode through the graph: stay and chat → teacher, ready → test, switch → confirm_switch.
def t_learn_routes() -> str:
    graph, config, db, models = learn_graph("learn-routes")
    teacher_calls = sum(kind == "teacher" for kind, _ in models.chat.calls)
    db.matches = matches(("L1.02", 0.8), ("L0.01", 0.4))
    snapshot, waiting = step(graph, config, {"type": "message", "text": "why is kataba joined?"})
    check(waiting["kind"] == "learn" and snapshot.values["route"]["fast_path"], "stay → teacher via fast path")
    db.matches = matches(("L0.01", 0.3))
    models.chat.replies["router"] = [router_json("chat")]
    snapshot, waiting = step(graph, config, {"type": "message", "text": "hello!"})
    check(waiting["kind"] == "learn" and snapshot.values["route"]["intent"] == "chat", "chat → teacher")
    check(sum(kind == "teacher" for kind, _ in models.chat.calls) == teacher_calls + 2, "teacher answered both")
    check("steer back to the lesson" in models.chat.last_system("teacher"), "teacher must steer back")
    models.chat.replies["router"] = [router_json("switch", "L4.03")]
    db.matches = matches(("L4.03", 0.7), ("L0.01", 0.4))
    snapshot, waiting = step(graph, config, {"type": "message", "text": "متى اكتب الهمزه على واو"})
    check(waiting["kind"] == "confirm_switch" and [o["lesson_id"] for o in waiting["options"]] == ["L4.03"],
          f"switch → confirm_switch: {waiting}")
    check("Shall we move" in str(snapshot.values["messages"][-1].content), "the question is asked in the chat")
    graph, config, db, models = learn_graph("learn-ready")
    models.chat.replies["router"] = [router_json("ready")]
    snapshot, waiting = step(graph, config, {"type": "message", "text": "خلاص اختبرني"})
    check(waiting["kind"] == "canvas" and snapshot.values["mode"] == "test", "ready → test")
    return "stay (fast) / chat → teacher, ready → test, switch → confirm"


# confirm_switch: yes (with the prerequisite note), no, a two-lesson choice, and a typed message.
def t_confirm_switch() -> str:
    graph, config, db, models = learn_graph("switch-yes")
    db.matches = matches(("L4.03", 0.7), ("L0.01", 0.4))
    models.chat.replies["router"] = [router_json("switch", "L4.03")]
    step(graph, config, {"type": "message", "text": "middle hamza please"})
    snapshot, waiting = step(graph, config, {"type": "switch_answer", "lesson_id": "L4.03"})
    values = snapshot.values
    check(waiting["kind"] == "learn" and values["lesson"]["lesson_id"] == "L4.03", "yes → new lesson")
    check(db.profiles["user-1"]["current_lesson_id"] == "L4.03", "the switch is saved to the profile")
    note = next(str(m.content) for m in reversed(values["messages"]) if "move to lesson" in str(m.content))
    check("L4.02" in note and "haven't passed" in note, f"prerequisite note expected: {note}")
    check(isinstance(values["messages"][-1], AIMessage) and "move to lesson" not in str(values["messages"][-1].content),
          "the teacher introduces the new lesson after the note")

    graph, config, db, models = learn_graph("switch-no")
    db.matches = matches(("L4.03", 0.7), ("L0.01", 0.4))
    models.chat.replies["router"] = [router_json("switch", "L4.03")]
    step(graph, config, {"type": "message", "text": "middle hamza please"})
    snapshot, waiting = step(graph, config, {"type": "switch_answer", "lesson_id": None})
    check(waiting["kind"] == "learn" and snapshot.values["lesson"]["lesson_id"] == "L1.02", "no → stay")
    check("What would you like to do?" in str(snapshot.values["messages"][-1].content), "the teacher asks what they want")

    graph, config, db, models = learn_graph("switch-two")
    db.matches = matches(("L4.03", 0.62), ("L0.01", 0.60), ("L1.02", 0.3))
    models.chat.replies["router"] = [router_json("switch", "L4.03", unsure=True)]
    snapshot, waiting = step(graph, config, {"type": "message", "text": "hamza or letters?"})
    check([o["lesson_id"] for o in waiting["options"]] == ["L4.03", "L0.01"], f"two options: {waiting}")
    check("Which lesson do you mean" in str(snapshot.values["messages"][-1].content), "asks which lesson")
    snapshot, waiting = step(graph, config, {"type": "switch_answer", "lesson_id": "L0.01"})
    check(snapshot.values["lesson"]["lesson_id"] == "L0.01", "the second option can be chosen")

    graph, config, db, models = learn_graph("switch-typed")
    db.matches = matches(("L4.03", 0.7), ("L0.01", 0.4))
    models.chat.replies["router"] = [router_json("switch", "L4.03"), router_json("chat")]
    step(graph, config, {"type": "message", "text": "middle hamza please"})
    snapshot, waiting = step(graph, config, {"type": "message", "text": "actually never mind"})
    check(waiting["kind"] == "learn" and snapshot.values["lesson"]["lesson_id"] == "L1.02", "typing declines and is routed")
    return "yes + prerequisite note, no, two-lesson choice, typed message"


# Test mode: stay / chat / ready → examiner, stop → learn without a score, switch → finish or stop.
def t_test_routes() -> str:
    graph, config, db, models = learn_graph("test-routes")
    step(graph, config, {"type": "ready"})
    for router_reply in (router_json("stay"), router_json("chat"), router_json("ready")):
        models.chat.replies["router"] = [router_reply]
        snapshot, waiting = step(graph, config, {"type": "message", "text": "how do I write?"})
        check(waiting["kind"] == "canvas" and waiting["index"] == 0, f"{router_reply} → back to the canvas")
    check("Never write, spell or hint" in models.chat.last_system("test_chat"), "the examiner must not give answers")
    models.ocr_outputs = ["كتب"]
    step(graph, config, canvas_payload(db))
    models.chat.replies["router"] = [router_json("chat")]
    snapshot, waiting = step(graph, config, {"type": "message", "text": "what does this mean?"})
    check(waiting["kind"] == "confirm_read", "a message during confirm returns there")
    models.chat.replies["router"] = [router_json("stop")]
    snapshot, waiting = step(graph, config, {"type": "message", "text": "وقف الاختبار ابي ارجع للدرس"})
    check(waiting["kind"] == "learn" and snapshot.values["mode"] == "learn", "stop → learn")
    check(snapshot.values["lesson"]["lesson_id"] == "L1.02" and not db.progress, "stop must not save a score")
    return "stay/chat/ready → examiner, stop → learn, no score"


# confirm_switch_mid_test: finish → test continues, switch offered after; stop → switch now, no score.
def t_confirm_switch_mid_test() -> str:
    graph, config, db, models = learn_graph("mid-finish")
    step(graph, config, {"type": "ready"})
    db.matches = matches(("L0.01", 0.7), ("L4.03", 0.4))
    models.chat.replies["router"] = [router_json("switch", "L0.01")]
    snapshot, waiting = step(graph, config, {"type": "message", "text": "ابي حرف الثاء بدال الاختبار"})
    check(waiting["kind"] == "confirm_switch_mid_test", f"switch in test → ask finish/stop: {waiting}")
    snapshot, waiting = step(graph, config, {"type": "test_switch_answer", "choice": "finish"})
    check(waiting["kind"] == "canvas" and waiting["index"] == 0, "finish → back to the same item")
    answer_item(graph, config, db, models)
    snapshot, waiting = answer_item(graph, config, db, models)
    check(("user-1", "L1.02") in db.progress, "the finished test is saved")
    check(waiting["kind"] == "confirm_switch" and [o["lesson_id"] for o in waiting["options"]] == ["L0.01"],
          f"the switch is offered again after the test: {waiting}")

    graph, config, db, models = learn_graph("mid-stop")
    step(graph, config, {"type": "ready"})
    answer_item(graph, config, db, models)
    db.matches = matches(("L4.03", 0.7), ("L0.01", 0.4))
    models.chat.replies["router"] = [router_json("switch", "L4.03")]
    step(graph, config, {"type": "message", "text": "stop, teach me middle hamza"})
    snapshot, waiting = step(graph, config, {"type": "test_switch_answer", "choice": "stop"})
    values = snapshot.values
    check(waiting["kind"] == "learn" and values["lesson"]["lesson_id"] == "L4.03", "stop → switched")
    check(values["mode"] == "learn" and not values["test_items"], "the test is cleared")
    check(not db.progress, "stop must not call save_test_result")
    return "finish → test, then offer L0.01; stop → L4.03, nothing saved"


# ---------- end-to-end walk ----------

# Resumes the graph and returns the new snapshot and pending interrupt value.
def step(graph: Any, config: dict[str, Any], payload: Any) -> tuple[Any, dict[str, Any] | None]:
    graph.invoke(Command(resume=payload) if payload is not None else None, config)
    snapshot = graph.get_state(config)
    return snapshot, pending_interrupt(snapshot)


# Uploads a fake PNG and returns the canvas resume payload.
def canvas_payload(db: FakeDatabase) -> dict[str, Any]:
    image_path, strokes_path = db.upload_attempt_files("user-1", b"\x89PNG", {"objects": [{"type": "path"}]})
    return {"type": "canvas", "image_path": image_path, "strokes_path": strokes_path, "reveal_count": 2}


# Full walk: onboarding → teacher → ready → 4 items (correct, wrong letter, wrong haraka, skipped) → result.
def t_end_to_end() -> str:
    db, models = FakeDatabase(), FakeModels()
    graph = build_graph(TutorNodes(db, models, test_size=4, seed=7), MemorySaver())
    config = {"configurable": {"thread_id": "user-1:e2e"}}

    graph.invoke({"user_id": "user-1", "messages": []}, config)
    snapshot = graph.get_state(config)
    waiting = pending_interrupt(snapshot)
    check(waiting["kind"] == "onboarding" and waiting["field"] == "learner_type", f"start: {waiting}")
    snapshot, waiting = step(graph, config, {"type": "message", "text": "non-native"})
    check(waiting["field"] == "explain_language", f"after type: {waiting}")
    snapshot, waiting = step(graph, config, {"type": "message", "text": "English"})
    check(waiting["field"] == "start_level", f"after language: {waiting}")
    snapshot, waiting = step(graph, config, {"type": "message", "text": "1"})
    values = snapshot.values
    check(waiting["kind"] == "learn" and values["lesson"]["lesson_id"] == "L1.02", "should be teaching L1.02")
    check(str(values["messages"][-1].content).endswith("?"), "teacher must end with readiness question")

    models.chat.replies["router"] = [router_json("ready")]
    snapshot, waiting = step(graph, config, {"type": "message", "text": "yes"})
    items = snapshot.values["test_items"]
    check(snapshot.values["mode"] == "test" and len(items) == 4 and waiting["kind"] == "canvas", "test should start")

    # Item 1: fully correct.
    models.ocr_outputs = [items[0]]
    snapshot, waiting = step(graph, config, canvas_payload(db))
    check(waiting["kind"] == "confirm_read" and waiting["ocr_text"] == strip_harakat(items[0]), f"confirm 1: {waiting}")
    snapshot, waiting = step(graph, config, {"type": "confirm_read", "ok": True})
    check(waiting["kind"] == "harakat", "level 1 right letters → harakat")
    right_marks = [{"shadda": e["shadda"], "vowel": e["vowel"]} for e in split_letters_and_marks(items[0])]
    snapshot, waiting = step(graph, config, {"type": "harakat", "marks": right_marks})
    check(snapshot.values["test_results"][0]["is_correct"] is True, "item 1 should be correct")
    check(waiting["kind"] == "canvas" and waiting["index"] == 1, "should move to item 2")

    # Item 2: wrong letter (first letter replaced).
    bare = strip_harakat(items[1])
    models.ocr_outputs = [("ث" if bare[0] != "ث" else "ت") + bare[1:]]
    step(graph, config, canvas_payload(db))
    snapshot, waiting = step(graph, config, {"type": "confirm_read", "ok": True})
    result = snapshot.values["test_results"][1]
    check(result["letters_correct"] is False and result["harakat_correct"] is None, f"item 2: {result}")
    check(waiting["index"] == 2, "wrong letters must skip the harakat step")

    # Item 3: right letters, one wrong haraka.
    models.ocr_outputs = [items[2]]
    step(graph, config, canvas_payload(db))
    step(graph, config, {"type": "confirm_read", "ok": True})
    wrong_marks = [{"shadda": e["shadda"], "vowel": e["vowel"]} for e in split_letters_and_marks(items[2])]
    wrong_marks[0]["vowel"] = "kasra"
    snapshot, waiting = step(graph, config, {"type": "harakat", "marks": wrong_marks})
    result = snapshot.values["test_results"][2]
    check(result["letters_correct"] and result["harakat_correct"] is False, f"item 3: {result}")

    # Item 4: disputed twice and skipped.
    models.ocr_outputs = ["ببب", "تتت"]
    step(graph, config, canvas_payload(db))
    snapshot, waiting = step(graph, config, {"type": "confirm_read", "ok": False})
    check(waiting["kind"] == "canvas" and waiting["index"] == 3 and waiting["attempt"] == 1, f"retry: {waiting}")
    step(graph, config, canvas_payload(db))
    snapshot, waiting = step(graph, config, {"type": "confirm_read", "ok": False})

    values = snapshot.values
    skipped = [a for a in db.attempts if a["target"] == items[3]]
    check(len(skipped) == 2 and not any(a["counted"] for a in skipped), "item 4 should be skipped (2 uncounted rows)")
    check(values["test_results"] == [], "load_lesson should clear the finished test")
    check(values["mode"] == "learn" and waiting["kind"] == "learn", "should be back in learn mode")
    progress = db.progress[("user-1", "L1.02")]
    check(abs(progress["best_score"] - 0.333) < 0.001 and progress["status"] == "in_progress", f"progress {progress}")
    check(values["lesson"]["lesson_id"] == "L1.02", "fail must stay on the lesson")
    check(len(values["review_items"]) == 2, f"review items {values['review_items']}")
    check(len(db.attempts) == 5 and sum(a["counted"] for a in db.attempts) == 3, "3 graded + 2 disputed attempts expected")
    return "score 1/3 = 33% → stays on L1.02"


# ---------- live tests ----------

# Reads L0.01 from the real Supabase.
def t_live_read_lesson() -> str:
    from Database import Database

    lesson = Database().get_lesson("L0.01")
    check(lesson is not None and lesson_items(lesson), "L0.01 missing or empty")
    return f"{len(lesson_items(lesson))} items"


# Finds an installed font that has Arabic glyphs.
def find_arabic_font() -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Baghdad.ttc",
        "/System/Library/Fonts/GeezaPro.ttc",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No Arabic font found; add one to find_arabic_font()")


# Renders Arabic text to PNG; shapes with raqm when available, else draws letters in visual order.
def render_arabic_png(text: str) -> bytes:
    from PIL import ImageDraw, ImageFont, features

    font = ImageFont.truetype(find_arabic_font(), 120)
    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    if features.check("raqm"):
        draw.text((450, 130), text, font=font, fill="black", anchor="mm", direction="rtl")
    else:
        draw.text((450, 130), text[::-1], font=font, fill="black", anchor="mm")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# Sends a rendered lesson word from Supabase to the real OCR.
def t_live_ocr() -> str:
    from Database import Database
    from model import Models

    word = lesson_items(Database().get_lesson("L1.02"))[0]["text"]
    read = Models().read_handwriting(render_arabic_png(strip_harakat(word)))
    check(bool(read), "OCR returned nothing")
    match = compare_letters(word, read)["letters_correct"]
    return f"target {strip_harakat(word)} · read {read} · match={match}"


# Sends one real chat call.
def t_live_chat() -> str:
    from model import Models

    reply = Models().chat.invoke([HumanMessage("Reply with the single word: pong")])
    check(bool(reply.text.strip()), "empty chat reply")
    return f"reply: {reply.text.strip()[:40]}"


# Writes and reads a checkpoint in Postgres, then deletes the test thread.
def t_live_checkpointer() -> str:
    db_url = os.environ.get("SUPABASE_DB_URL")
    check(bool(db_url), "SUPABASE_DB_URL not set")
    saver, _ = make_checkpointer(db_url)
    graph = build_graph(TutorNodes(FakeDatabase(), FakeModels()), saver)
    config = {"configurable": {"thread_id": f"test_graph:{uuid.uuid4().hex}"}}
    try:
        graph.invoke({"user_id": "user-1", "messages": []}, config)
        snapshot = graph.get_state(config)
        check(pending_interrupt(snapshot) is not None, "checkpoint not read back")
    finally:
        saver.delete_thread(config["configurable"]["thread_id"])
    return "write + read + delete ok"


# ---------- runner ----------

UNIT_STEPS: list[tuple[str, Callable[[], str]]] = [
    ("load_session (new)", t_load_session_new),
    ("load_session (returning)", t_load_session_returning),
    ("ask_field", t_ask_field),
    ("onboarding: child", t_onboarding_child),
    ("onboarding: language", t_onboarding_language),
    ("onboarding: start level", t_onboarding_level),
    ("onboarding: invalid", t_onboarding_invalid),
    ("load_lesson", t_load_lesson),
    ("teacher", t_teacher),
    ("route: fast path", t_fast_path),
    ("route: parse / timeout / NEXT", t_parse_route),
    ("route: switch options", t_switch_options),
    ("route: table (learn × test)", t_route_table),
    ("card text", t_card_text),
    ("build_test_items", t_build_test_items),
    ("compare_letters", t_compare_letters),
    ("compare_harakat", t_compare_harakat),
    ("allowed_marks_for", t_allowed_marks),
    ("canvas_to_png", t_canvas_png),
    ("system prompt (L4)", t_system_prompt_rules),
    ("read_letters + yes", t_read_confirm_yes),
    ("confirm_read: no once", t_confirm_no_once),
    ("confirm_read: no twice", t_confirm_no_twice),
    ("check_letters routing", t_check_letters_routing),
    ("ask_harakat + grade_item", t_grade_item),
    ("finish_test", t_finish_test),
    ("learn routes (graph)", t_learn_routes),
    ("confirm_switch", t_confirm_switch),
    ("test routes (graph)", t_test_routes),
    ("confirm_switch_mid_test", t_confirm_switch_mid_test),
    ("end-to-end walk", t_end_to_end),
]

# ---------- live routing evaluation ----------

EVAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "routing_eval.jsonl")
INTENTS = ["stay", "switch", "ready", "chat", "stop"]
EVAL_RESULTS: list[dict[str, Any]] = []


# Reads data/routing_eval.jsonl, one case per line.
def read_eval_cases() -> list[dict[str, Any]]:
    with open(EVAL_FILE, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# True when the routed intent matches; vague cases also accept "switch, unsure" to any lesson.
def intent_correct(case: dict[str, Any], decision: dict[str, Any]) -> bool:
    if decision["intent"] != case["expected_intent"]:
        return False
    if case["expected_intent"] != "switch":
        return True
    return case["expected_lesson_id"] in decision["options"] or bool(case.get("accept_unsure") and decision.get("unsure"))


# Routes one case through the real route_message node and records what happened.
def evaluate_case(nodes: TutorNodes, course: dict[str, dict[str, Any]], case: dict[str, Any]) -> dict[str, Any]:
    state = {"user_id": "eval", "profile": {"learner_type": "child", "explain_language": "ar"},
             "lesson": course[case["current_lesson_id"]], "mode": case.get("mode", "learn"),
             "messages": [HumanMessage(case["message"])]}
    started = time.perf_counter()
    decision = nodes.route_message(state)["route"]
    seconds = time.perf_counter() - started
    retrieved = [match["lesson_id"] for match in decision["candidates"]]
    target = case["expected_lesson_id"] if case["expected_intent"] == "switch" else case["current_lesson_id"]
    return {**case, "intent": decision["intent"], "options": decision["options"], "unsure": decision.get("unsure"),
            "fast_path": decision["fast_path"], "retrieved": retrieved, "seconds": seconds,
            "top1_similarity": decision["candidates"][0]["similarity"] if decision["candidates"] else 0.0,
            "retrieval_target": target, "top1_ok": retrieved[:1] == [target], "top3_ok": target in retrieved[:3],
            "intent_ok": intent_correct(case, decision)}


# Prints accuracy, the confusion matrix, latency, and every failed case.
def print_eval_report(results: list[dict[str, Any]]) -> dict[str, float]:
    switches = [r for r in results if r["expected_intent"] == "switch" and not r.get("accept_unsure")]
    seconds = np.array([r["seconds"] for r in results])
    numbers = {
        "top-1 retrieval": np.mean([r["top1_ok"] for r in switches]),
        "top-3 retrieval": np.mean([r["top3_ok"] for r in switches]),
        "intent accuracy": np.mean([r["intent_ok"] for r in results]),
        "fast-path rate": np.mean([r["fast_path"] for r in results]),
    }
    print(f"\nRouting evaluation: {len(results)} cases ({len(switches)} switch cases with a named topic)")
    for name, value in numbers.items():
        print(f"  {name:16} {value:6.1%}")
    print(f"  latency          median {np.median(seconds):.2f} s | p95 {np.percentile(seconds, 95):.2f} s")
    print("\nConfusion matrix (rows = expected, columns = routed):")
    print("          " + "".join(f"{name:>8}" for name in INTENTS))
    for expected in INTENTS:
        row = [sum(1 for r in results if r["expected_intent"] == expected and r["intent"] == got) for got in INTENTS]
        print(f"  {expected:8}" + "".join(f"{count:8d}" for count in row))
    failed = [r for r in results if not r["intent_ok"]]
    print(f"\nFailed cases ({len(failed)}):")
    print(f"  {'mode':5} {'current':7} {'expected':16} {'got':22} {'top-3 retrieved':24} message")
    for r in failed:
        expected = f"{r['expected_intent']} {r['expected_lesson_id'] or ''}".strip()
        got = f"{r['intent']} {','.join(r['options'])}{' (unsure)' if r['unsure'] else ''}".strip()
        print(f"  {r.get('mode', 'learn'):5} {r['current_lesson_id']:7} {expected:16} {got:22} "
              f"{','.join(r['retrieved']):24} {r['message']}")
    return {name: float(value) for name, value in numbers.items()}


# Runs every routing_eval.jsonl case against the real embeddings, Supabase and router.
def t_live_routing_eval() -> str:
    from Database import Database
    from model import Models

    nodes = TutorNodes(Database(), Models())
    course = {lesson["lesson_id"]: lesson for lesson in nodes.db.get_course()}
    EVAL_RESULTS.clear()
    EVAL_RESULTS.extend(evaluate_case(nodes, course, case) for case in read_eval_cases())
    numbers = print_eval_report(EVAL_RESULTS)
    summary = (f"top-3 {numbers['top-3 retrieval']:.0%} (target 95%), intent {numbers['intent accuracy']:.0%} "
               f"(target 90%), fast path {numbers['fast-path rate']:.0%}")
    check(numbers["top-3 retrieval"] >= 0.95 and numbers["intent accuracy"] >= 0.90, f"below target: {summary}")
    return summary


LIVE_STEPS: list[tuple[str, Callable[[], str]]] = [
    ("live: read L0.01", t_live_read_lesson),
    ("live: OCR", t_live_ocr),
    ("live: chat", t_live_chat),
    ("live: checkpointer", t_live_checkpointer),
    ("live: routing eval", t_live_routing_eval),
]


# Runs one step, timing it and catching any failure.
def run_step(name: str, function: Callable[[], str]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        detail, passed = function(), True
    except Exception as error:
        detail, passed = f"{type(error).__name__}: {error}", False
        if os.environ.get("TEST_VERBOSE"):
            traceback.print_exc()
    return {"step": name, "passed": passed, "ms": (time.perf_counter() - started) * 1000, "detail": detail}


# Prints the step | PASS/FAIL | ms | detail table.
def print_report(results: list[dict[str, Any]]) -> None:
    width = max(len(r["step"]) for r in results)
    print(f"{'step'.ljust(width)} | result | {'ms':>8} | detail")
    print("-" * (width + 40))
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['step'].ljust(width)} | {status:<6} | {r['ms']:8.1f} | {r['detail']}")
    failed = sum(not r["passed"] for r in results)
    print(f"\n{len(results) - failed} passed, {failed} failed")


# Draws time per step as horizontal bars, green PASS / red FAIL, each bar labelled.
def plot_report(results: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    good, critical, surface, grid, ink, muted = "#0ca30c", "#d03b3b", "#fcfcfb", "#e1e0d9", "#0b0b0b", "#52514e"
    names = [r["step"] for r in results][::-1]
    times = [r["ms"] for r in results][::-1]
    colors = [good if r["passed"] else critical for r in results][::-1]
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(results) + 1.5), facecolor=surface)
    ax.set_facecolor(surface)
    bars = ax.barh(names, times, color=colors, height=0.6, edgecolor=surface, linewidth=2)
    for bar, result in zip(bars, results[::-1]):
        label = f"{'PASS' if result['passed'] else 'FAIL'} · {result['ms']:.0f} ms"
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {label}", va="center", fontsize=8, color=muted)
    ax.set_xlabel("time per step (ms)", color=muted)
    ax.set_title("test_graph.py — time per step", color=ink, loc="left")
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(handles=[Patch(color=good, label="PASS"), Patch(color=critical, label="FAIL")], frameon=False, loc="lower right")
    ax.set_xlim(0, max(times) * 1.3 if times else 1)
    fig.tight_layout()


# Draws the routing evaluation: confusion heatmap, accuracy bars, and top-1 similarity histogram.
def plot_eval(results: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    good, critical, blue = "#0ca30c", "#d03b3b", "#2a6fdb"
    fig, (heat, bars, hist) = plt.subplots(1, 3, figsize=(16, 5), facecolor=surface)
    matrix = np.array([[sum(1 for r in results if r["expected_intent"] == e and r["intent"] == g) for g in INTENTS]
                       for e in INTENTS])
    heat.imshow(matrix, cmap="Blues")
    heat.set_xticks(range(len(INTENTS)), INTENTS)
    heat.set_yticks(range(len(INTENTS)), INTENTS)
    heat.set_xlabel("routed", color=muted)
    heat.set_ylabel("expected", color=muted)
    for i in range(len(INTENTS)):
        for j in range(len(INTENTS)):
            heat.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else ink, fontsize=10)
    heat.set_title("Intent confusion matrix", color=ink, loc="left")
    switches = [r for r in results if r["expected_intent"] == "switch" and not r.get("accept_unsure")]
    names = ["top-1 retrieval", "top-3 retrieval", "intent accuracy", "fast-path rate"]
    values = [np.mean([r["top1_ok"] for r in switches]), np.mean([r["top3_ok"] for r in switches]),
              np.mean([r["intent_ok"] for r in results]), np.mean([r["fast_path"] for r in results])]
    targets = [None, 0.95, 0.90, None]
    bars.set_facecolor(surface)
    drawn = bars.bar(names, values, color=blue, width=0.6, edgecolor=surface, linewidth=2)
    for bar, value, target in zip(drawn, values, targets):
        bars.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.0%}", ha="center", color=ink, fontsize=9)
        if target is not None:
            bars.hlines(target, bar.get_x(), bar.get_x() + bar.get_width(), colors=critical, linestyles="--", linewidth=1.5)
    bars.set_ylim(0, 1.1)
    bars.set_title("Accuracy (dashed = target)", color=ink, loc="left")
    bars.tick_params(axis="x", labelsize=8, colors=muted)
    bars.grid(axis="y", color=grid)
    bars.set_axisbelow(True)
    hist.set_facecolor(surface)
    topical = [r for r in results if r["expected_intent"] in ("switch", "stay") and not r.get("accept_unsure")]
    right = [r["top1_similarity"] for r in topical if r["top1_ok"]]
    wrong = [r["top1_similarity"] for r in topical if not r["top1_ok"]]
    bins = np.linspace(min(right + wrong + [0.3]), max(right + wrong + [0.8]), 16)
    hist.hist(right, bins=bins, color=good, alpha=0.75, label=f"correct top-1 ({len(right)})", edgecolor=surface)
    hist.hist(wrong, bins=bins, color=critical, alpha=0.75, label=f"wrong top-1 ({len(wrong)})", edgecolor=surface)
    hist.axvline(MIN_SIM, color=ink, linestyle="--", linewidth=1.2, label=f"MIN_SIM {MIN_SIM}")
    hist.set_xlabel("top-1 similarity (switch target, or current lesson for the rest)", color=muted)
    hist.set_title("Top-1 similarity: correct vs wrong retrievals", color=ink, loc="left")
    hist.legend(frameon=False, fontsize=8)
    for ax in (bars, hist):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()


# Parses flags, runs the steps, prints the report, plots it, and exits 1 on any failure.
def main() -> None:
    parser = argparse.ArgumentParser(description="Test every step of the Arabic tutor graph.")
    parser.add_argument("--live", action="store_true", help="also test Supabase, Cloudflare and Postgres")
    parser.add_argument("--no-plot", action="store_true", help="skip the matplotlib chart")
    args = parser.parse_args()
    if not args.live:
        os.environ["LANGSMITH_TRACING"] = "false"
    steps = UNIT_STEPS + (LIVE_STEPS if args.live else [])
    results = [run_step(name, function) for name, function in steps]
    print_report(results)
    if not args.no_plot:
        import matplotlib.pyplot as plt

        plot_report(results)
        if EVAL_RESULTS:
            plot_eval(EVAL_RESULTS)
        plt.show()
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
