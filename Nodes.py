"""Every LangGraph node and router of the tutor, as methods of one class with injected db and models."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langsmith import traceable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.types import interrupt

from tools import (
    ROUTER_PROMPT,
    ROUTER_TIMEOUT_SECONDS,
    build_test_chat_prompt,
    is_fast_stay,
    message_language,
    parse_route,
    router_request,
    switch_options,
    allowed_marks_for,
    build_feedback_prompt,
    build_system_prompt,
    build_test_items,
    compare_harakat,
    compare_letters,
    ends_with_question,
    harakat_tested,
    is_language_code,
    language_name,
    parse_language_code,
    parse_learner_type,
    parse_start_level,
    MESSAGES,
    learner_language,
    merge_translation,
    parse_json_object,
    say,
    split_letters_and_marks,
)

MAX_HISTORY = 20
ROUTER_POOL = ThreadPoolExecutor(max_workers=4)


# Records a routing decision as its own LangSmith run, with the decision as metadata for filtering.
@traceable(name="route_decision", run_type="chain")
def log_route_decision(decision: dict[str, Any]) -> dict[str, Any]:
    return decision


class TutorState(MessagesState):
    user_id: str
    profile: dict             # learner_type, explain_language, start_level, current_lesson_id, reveal_seconds
    missing_fields: list[str] # profile fields still to ask
    lesson: dict | None       # current lesson incl. rules
    mode: str                 # "learn" | "test"
    test_items: list[str]     # targets for this test (with harakat)
    test_index: int
    current_answer: dict      # in-progress item: ocr_text, dispute_count, letters_correct, image_path, strokes_path, reveal_count
    test_results: list[dict]  # per item: target, ocr_text, letters_correct, harakat_correct, is_correct, counted, errors
    onboarding_note: str      # why the last onboarding answer was rejected, so the next question can say so
    intent: str               # "READY" from the ready button; internal markers after the switch interrupts
    review_items: list[str]   # items answered wrong in the last failed test, for the teacher to review
    test_message: str         # a chat message typed during the test, waiting for test_chat
    resume_step: str          # the test step to return to after answering that message
    route: dict               # last routing decision: intent, lesson_id, unsure, candidates, options, fast_path
    switch_offer: dict        # lessons offered to switch to: options, pending (after the test), confirmed
    switch_choice: str        # lesson id the learner chose to switch to, "" if none


# Lists the profile fields onboarding still has to ask, in asking order.
def missing_profile_fields(profile: dict[str, Any]) -> list[str]:
    missing = []
    if profile.get("learner_type") not in ("child", "non_native"):
        missing.append("learner_type")
    if profile.get("learner_type") == "non_native" and not profile.get("explain_language"):
        missing.append("explain_language")
    if profile.get("start_level") is None:
        missing.append("start_level")
    return missing


# Returns a fresh in-progress answer for a test item; attempt numbers keep canvases unique.
def new_answer(attempt: int = 0, dispute_count: int = 0) -> dict[str, Any]:
    return {
        "ocr_text": None,
        "dispute_count": dispute_count,
        "ocr_disputed": dispute_count > 0,
        "confirmed": None,
        "letters_correct": None,
        "errors": [],
        "harakat_input": None,
        "image_path": None,
        "strokes_path": None,
        "reveal_count": 1,
        "attempt": attempt,
    }


class TutorNodes:
    # Stores the injected database and models; nodes never create clients themselves.
    def __init__(self, db: Any, models: Any, test_size: int = 5, seed: int | None = None) -> None:
        self.db = db
        self.models = models
        self.test_size = test_size
        self.seed = seed
        self.translations: dict[tuple[str, str], dict[str, str]] = {}

    # Translates a table of short UI strings into a language once, keeping {placeholders}; English fallback.
    def translate_table(self, language_code: str, table: dict[str, str], name: str = "ui") -> dict[str, str]:
        cache_key = (language_code, name)
        if cache_key not in self.translations:
            prompt = [
                SystemMessage(f"Translate every value of this JSON object into {language_name(language_code)}. "
                              "Keep the keys, keep {placeholders} unchanged, keep symbols like → and ✎. "
                              "Return only the JSON object."),
                HumanMessage(json.dumps(table, ensure_ascii=False)),
            ]
            try:
                reply = self.models.chat.invoke(prompt).text
                self.translations[cache_key] = merge_translation(table, parse_json_object(reply))
            except Exception:
                return table
        return self.translations[cache_key]

    # Returns a fixed message in the learner's language (Arabic, English, or translated from English).
    def _say(self, profile: dict[str, Any], key: str, **values: Any) -> str:
        language = learner_language(profile)
        if language in ("ar", "en"):
            return say(profile, key, **values)
        return self.translate_table(language, MESSAGES["en"], "messages")[key].format(**values)

    # ---------- session and onboarding ----------

    # Reads the profile, defaults children to Arabic explanations, and lists missing fields.
    def load_session(self, state: TutorState) -> dict[str, Any]:
        user_id = state["user_id"]
        profile = self.db.get_profile(user_id) or {}
        if profile.get("learner_type") == "child" and not profile.get("explain_language"):
            profile = self.db.upsert_profile(user_id, explain_language="ar")
        return {"profile": profile, "missing_fields": missing_profile_fields(profile), "onboarding_note": ""}

    # Routes to onboarding while fields are missing, otherwise to the lesson.
    def route_after_session(self, state: TutorState) -> str:
        return "ask_field" if state.get("missing_fields") else "load_lesson"

    # Has the chat model ask one friendly question about the next missing field.
    def ask_field(self, state: TutorState) -> dict[str, Any]:
        field = state["missing_fields"][0]
        instruction = self._field_question_instruction(field, state.get("profile") or {})
        note = state.get("onboarding_note") or ""
        prompt = [
            SystemMessage(f"You are welcoming a new learner to an Arabic spelling tutor. {instruction} {note} "
                          "Ask only this one short, friendly question. Do not teach anything yet."),
            HumanMessage("Ask the question now."),
        ]
        reply = self.models.chat.invoke(prompt)
        return {"messages": [AIMessage(reply.text)]}

    # Describes what to ask for each onboarding field, and in which language.
    def _field_question_instruction(self, field: str, profile: dict[str, Any]) -> str:
        if field == "learner_type":
            return ("Ask, in Arabic first and then English, whether the learner is an Arabic-speaking child "
                    "or a non-native speaker learning Arabic.")
        if field == "explain_language":
            return "Ask, in English, which language they want explanations in (for example English, Urdu or Indonesian)."
        language = "Arabic" if profile.get("learner_type") == "child" else language_name(profile.get("explain_language"))
        levels = "; ".join(f"{row['level']}: {row['title_en']} / {row['title_ar']}" for row in self.db.get_levels())
        return (f"Ask, in {language}, which level they want to start at. Offer these levels: {levels}. "
                "Say that 'I don't know' is fine and starts at level 0.")

    # Waits for the learner's answer, validates it, and saves it to the profile.
    def onboarding(self, state: TutorState) -> dict[str, Any]:
        field = state["missing_fields"][0]
        payload = interrupt({"kind": "onboarding", "field": field, "options": self._field_options(field)})
        text = str(payload.get("text") or "").strip() if isinstance(payload, dict) else ""
        human = HumanMessage(text or "…")
        value = self._parse_field(field, text)
        if value is None:
            note = f"The learner's previous answer '{text}' could not be understood; kindly ask again."
            return {"messages": [human], "onboarding_note": note}
        fields: dict[str, Any] = {field: value}
        if field == "learner_type" and value == "child":
            fields["explain_language"] = "ar"
        profile = self.db.upsert_profile(state["user_id"], **fields)
        return {"messages": [human], "profile": profile,
                "missing_fields": missing_profile_fields(profile), "onboarding_note": ""}

    # Quick-reply options the UI shows as buttons for a field.
    def _field_options(self, field: str) -> list[dict[str, str]]:
        if field == "learner_type":
            return [{"label": "طفل / Child", "text": "child"},
                    {"label": "Non-native speaker", "text": "non-native"}]
        if field == "start_level":
            options = [{"label": f"{row['level']} · {row['title_en']} · {row['title_ar']}", "text": str(row["level"])}
                       for row in self.db.get_levels()]
            return options + [{"label": "I don't know / لا أعرف", "text": "I don't know"}]
        return []

    # Turns a free-text answer into a valid field value, or None when invalid.
    def _parse_field(self, field: str, text: str) -> Any:
        if field == "learner_type":
            return parse_learner_type(text)
        if field == "start_level":
            return parse_start_level(text)
        return parse_language_code(text) or self._language_code_from_llm(text)

    # Asks the chat model for an ISO 639-1 code when the language name is not in our list.
    def _language_code_from_llm(self, text: str) -> str | None:
        if not text:
            return None
        prompt = [
            SystemMessage("Return only the ISO 639-1 two-letter code of the language the user names. "
                          "If no language is named, return XX. Output nothing else."),
            HumanMessage(text),
        ]
        code = self.models.chat.invoke(prompt).text.strip().lower()
        return code if is_language_code(code) and code != "xx" else None

    # Routes back to the next question until every field is filled.
    def route_after_onboarding(self, state: TutorState) -> str:
        return "ask_field" if state.get("missing_fields") else "load_lesson"

    # Writes "transliteration — meaning" for a word card shown to non-native learners (not a graph node).
    def gloss_item(self, text: str, meaning_en: str | None, language_code: str | None) -> str:
        language = language_name(language_code)
        meaning_hint = f"Its English meaning is: {meaning_en}." if meaning_en else ""
        prompt = [
            SystemMessage(f"Give the Latin transliteration of the Arabic text, then its meaning in {language}. "
                          f"{meaning_hint} Format: transliteration — meaning. Output only that one line."),
            HumanMessage(text),
        ]
        return self.models.chat.invoke(prompt).text.strip()

    # ---------- learning ----------

    # Loads the current lesson, or the first lesson of the start level, and enters learn mode.
    def load_lesson(self, state: TutorState) -> dict[str, Any]:
        profile = dict(state.get("profile") or {})
        lesson_id = profile.get("current_lesson_id")
        lesson = self.db.get_lesson(lesson_id) if lesson_id else None
        if lesson is None:
            lesson = self.db.get_first_lesson_of_level(int(profile.get("start_level") or 0))
            profile = self.db.upsert_profile(state["user_id"], current_lesson_id=lesson["lesson_id"])
        offer = dict(state.get("switch_offer") or {})
        if offer.get("pending"):
            offer["options"] = [lesson_id for lesson_id in offer.get("options") or [] if lesson_id != lesson["lesson_id"]]
        if not offer.get("pending") or not offer.get("options"):
            offer = {}
        return {"lesson": lesson, "profile": profile, "mode": "learn", "test_items": [],
                "test_index": 0, "current_answer": {}, "test_results": [], "switch_offer": offer}

    # After loading: offer a switch postponed until the test ended, otherwise teach.
    def route_after_load(self, state: TutorState) -> str:
        return "offer_switch" if (state.get("switch_offer") or {}).get("pending") else "teacher"

    # Explains the lesson in the learner's style; every reply ends with the readiness question.
    def teacher(self, state: TutorState) -> dict[str, Any]:
        profile = state.get("profile") or {}
        system = build_system_prompt(profile, state["lesson"], state.get("review_items"))
        history = self._chat_history(state.get("messages") or [])
        reply = self.models.chat.invoke([SystemMessage(system)] + history)
        text = reply.text.strip()
        if not ends_with_question(text):
            text = f"{text}\n\n{self._say(profile, 'ready_question')}"
        return {"messages": [AIMessage(text)]}

    # Keeps recent human/AI turns and makes sure the model is answering a human turn.
    def _chat_history(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        history = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][-MAX_HISTORY:]
        if not history or not isinstance(history[-1], HumanMessage):
            history.append(HumanMessage("(Teach this lesson now, or continue teaching it.)"))
        return history

    # Waits for a typed message or the "I'm ready" button.
    def wait_for_learner(self, state: TutorState) -> dict[str, Any]:
        payload = interrupt({"kind": "learn"})
        if isinstance(payload, dict) and payload.get("type") == "ready":
            return {"intent": "READY"}
        text = str(payload.get("text") or "").strip() if isinstance(payload, dict) else ""
        return {"messages": [HumanMessage(text or "…")], "intent": ""}

    # The ready button goes straight to the test; typed messages go to the router.
    def route_after_wait(self, state: TutorState) -> str:
        return "build_test" if state.get("intent") == "READY" else "route_message"

    # ---------- routing (RAG over lesson cards) ----------

    # Returns the text of the learner's latest message.
    def _last_human_text(self, state: TutorState) -> str:
        last_human = next((m for m in reversed(state.get("messages") or []) if isinstance(m, HumanMessage)), None)
        return str(last_human.content).strip() if last_human else ""

    # Embeds the message and returns the 3 closest lessons; an empty list if retrieval fails.
    def _retrieve(self, text: str) -> list[dict[str, Any]]:
        try:
            return self.db.match_lessons(self.models.embed([text])[0], k=3)
        except Exception:
            return []

    # Returns the course lessons by id, in course order.
    def _course_by_id(self) -> dict[str, dict[str, Any]]:
        return {lesson["lesson_id"]: lesson for lesson in self.db.get_course()}

    # Builds router candidates: the retrieved lessons plus the next lesson in the course, marked NEXT.
    def _candidates(self, current_id: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        course = self._course_by_id()
        candidates = [{"lesson": course[m["lesson_id"]], "similarity": m["similarity"]}
                      for m in matches if m["lesson_id"] in course]
        order = list(course)
        if current_id in order and order.index(current_id) + 1 < len(order):
            next_id = order[order.index(current_id) + 1]
            existing = next((c for c in candidates if c["lesson"]["lesson_id"] == next_id), None)
            if existing:
                existing["marker"] = "NEXT"
            else:
                candidates.append({"lesson": course[next_id], "similarity": None, "marker": "NEXT"})
        return candidates

    # One strict router call with a timeout; any failure or timeout means "stay".
    def _ask_router(self, text: str, mode: str, lesson: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = self._candidates(lesson["lesson_id"], matches)
        request = router_request(text, mode, lesson, candidates)
        call = ROUTER_POOL.submit(self.models.chat.invoke, [SystemMessage(ROUTER_PROMPT), HumanMessage(request)])
        try:
            reply = call.result(timeout=ROUTER_TIMEOUT_SECONDS)
        except Exception as error:
            call.cancel()
            return {"intent": "stay", "lesson_id": None, "unsure": False, "valid": False, "error": type(error).__name__}
        return parse_route(reply.text, [c["lesson"]["lesson_id"] for c in candidates])

    # Routes one chat message: retrieval, then the fast path or the router; logs the decision.
    def route_message(self, state: TutorState) -> dict[str, Any]:
        started = time.perf_counter()
        text = self._last_human_text(state)
        lesson = state["lesson"]
        mode = state.get("mode") or "learn"
        matches = self._retrieve(text)
        fast_path = is_fast_stay(matches, lesson["lesson_id"])
        if fast_path:
            route = {"intent": "stay", "lesson_id": None, "unsure": False, "valid": True}
        else:
            route = self._ask_router(text, mode, lesson, matches)
        options = switch_options(route, matches, lesson["lesson_id"]) if route["intent"] == "switch" else []
        if route["intent"] == "switch" and not options:
            route = {**route, "intent": "stay"}
        decision = {**route, "mode": mode, "current_lesson_id": lesson["lesson_id"], "fast_path": fast_path,
                    "candidates": matches, "options": options, "seconds": round(time.perf_counter() - started, 3)}
        log_route_decision(decision, langsmith_extra={"metadata": {
            "route_intent": decision["intent"], "fast_path": fast_path, "route_options": options,
            "candidates": [f"{m['lesson_id']}:{m['similarity']:.3f}" for m in matches]}})
        update: dict[str, Any] = {"route": decision, "intent": "", "test_message": ""}
        if decision["intent"] == "switch":
            update["switch_offer"] = {"options": options, "pending": False, "confirmed": False}
        return update

    # Learn mode: switch → confirm, ready → test, anything else → teacher. Test mode: see the table in README.
    def route_after_message(self, state: TutorState) -> str:
        intent = (state.get("route") or {}).get("intent")
        if state.get("mode") == "test":
            return {"switch": "offer_test_switch", "stop": "exit_test"}.get(intent, "test_chat")
        return {"ready": "build_test", "switch": "offer_switch"}.get(intent, "teacher")

    # Returns a lesson's title in the learner's message language.
    def _title(self, profile: dict[str, Any], lesson_id: str) -> str:
        lesson = self._course_by_id().get(lesson_id) or {}
        title = lesson.get("title_ar") if message_language(profile) == "ar" else lesson.get("title_en")
        return title or lesson_id

    # Lists the offered lessons as {lesson_id, title} for the UI buttons.
    def _option_info(self, profile: dict[str, Any], options: list[str]) -> list[dict[str, str]]:
        return [{"lesson_id": lesson_id, "title": self._title(profile, lesson_id)} for lesson_id in options]

    # Asks whether to switch (one option) or which lesson is meant (two options).
    def offer_switch(self, state: TutorState) -> dict[str, Any]:
        profile = state.get("profile") or {}
        offer = dict(state.get("switch_offer") or {})
        titles = [self._title(profile, lesson_id) for lesson_id in offer.get("options") or []]
        if len(titles) >= 2:
            text = self._say(profile, "offer_switch_two", first=titles[0], second=titles[1])
        else:
            text = self._say(profile, "offer_switch_one", title=titles[0] if titles else "")
        return {"messages": [AIMessage(text)], "switch_offer": {**offer, "pending": False}}

    # Waits for the Yes / No (or lesson / Neither) buttons; a typed message declines and is routed instead.
    def confirm_switch(self, state: TutorState) -> dict[str, Any]:
        profile = state.get("profile") or {}
        options = (state.get("switch_offer") or {}).get("options") or []
        payload = interrupt({"kind": "confirm_switch", "options": self._option_info(profile, options)})
        if self._is_chat(payload):
            return {"messages": [HumanMessage(str(payload["text"]).strip())], "switch_offer": {},
                    "switch_choice": "", "intent": "MESSAGE"}
        if not (isinstance(payload, dict) and payload.get("type") == "switch_answer"):
            return {"intent": "REASK"}
        chosen = payload.get("lesson_id")
        return {"switch_choice": chosen if chosen in options else "", "switch_offer": {}, "intent": ""}

    # A chosen lesson is applied, "No"/"Neither" stays, a typed message is routed, bad payloads re-ask.
    def route_after_confirm_switch(self, state: TutorState) -> str:
        if state.get("intent") == "MESSAGE":
            return "route_message"
        if state.get("intent") == "REASK":
            return "confirm_switch"
        return "apply_switch" if state.get("switch_choice") else "decline_switch"

    # Saves the new current lesson, loads it, and notes unpassed prerequisites; progress on the old lesson stays.
    def apply_switch(self, state: TutorState) -> dict[str, Any]:
        profile = dict(state.get("profile") or {})
        lesson = self.db.get_lesson(state.get("switch_choice") or "")
        if lesson is None:
            return {"switch_choice": "", "switch_offer": {}}
        self.db.set_current_lesson(state["user_id"], lesson["lesson_id"])
        profile["current_lesson_id"] = lesson["lesson_id"]
        title = self._title(profile, lesson["lesson_id"])
        note = self._say(profile, "switched", lesson_id=lesson["lesson_id"], title=title)
        missing = self._unpassed_prerequisites(state["user_id"], lesson)
        if missing:
            titles = "، ".join(self._title(profile, lesson_id) for lesson_id in missing)
            note = f"{note} {self._say(profile, 'switch_prereq', titles=titles)}"
        return {"lesson": lesson, "profile": profile, "mode": "learn", "test_items": [], "test_index": 0,
                "current_answer": {}, "test_results": [], "review_items": [], "switch_choice": "",
                "switch_offer": {}, "resume_step": "", "test_message": "", "messages": [AIMessage(note)]}

    # Returns the lesson's prerequisites that the learner has not passed yet.
    def _unpassed_prerequisites(self, user_id: str, lesson: dict[str, Any]) -> list[str]:
        prerequisites = lesson.get("prerequisites") or []
        if isinstance(prerequisites, str):
            prerequisites = json.loads(prerequisites or "[]")
        passed = {row["lesson_id"] for row in self.db.get_progress(user_id) if row.get("status") == "passed"}
        return [lesson_id for lesson_id in prerequisites if lesson_id not in passed]

    # Stays on the current lesson and asks what the learner wants.
    def decline_switch(self, state: TutorState) -> dict[str, Any]:
        profile = state.get("profile") or {}
        title = self._title(profile, state["lesson"]["lesson_id"])
        return {"switch_choice": "", "switch_offer": {},
                "messages": [AIMessage(self._say(profile, "switch_declined", title=title))]}

    # During a test: asks whether to finish the test first or stop now and switch.
    def offer_test_switch(self, state: TutorState) -> dict[str, Any]:
        profile = state.get("profile") or {}
        options = (state.get("switch_offer") or {}).get("options") or []
        title = " / ".join(self._title(profile, lesson_id) for lesson_id in options)
        return {"messages": [AIMessage(self._say(profile, "offer_test_switch", title=title))]}

    # Waits for Finish / Stop; a typed message is routed and the question is asked again afterwards.
    def confirm_switch_mid_test(self, state: TutorState) -> dict[str, Any]:
        profile = state.get("profile") or {}
        offer = dict(state.get("switch_offer") or {})
        payload = interrupt({"kind": "confirm_switch_mid_test",
                             "options": self._option_info(profile, offer.get("options") or [])})
        if self._is_chat(payload):
            text = str(payload["text"]).strip()
            return {"messages": [HumanMessage(text)], "test_message": text, "switch_offer": {}, "intent": "MESSAGE"}
        choice = payload.get("choice") if isinstance(payload, dict) and payload.get("type") == "test_switch_answer" else None
        if choice == "finish":
            return {"switch_offer": {**offer, "pending": True}, "intent": "FINISH",
                    "messages": [AIMessage(self._say(profile, "finish_first"))]}
        if choice == "stop":
            return {"switch_offer": {**offer, "confirmed": True}, "intent": "STOP"}
        return {"intent": "REASK"}

    # Finish → back to the test step; Stop → leave the test; message → router; bad payload → ask again.
    def route_after_test_switch(self, state: TutorState) -> str:
        intent = state.get("intent")
        if intent == "MESSAGE":
            return "route_message"
        if intent == "FINISH":
            return state.get("resume_step") or "present_item"
        if intent == "STOP":
            return "exit_test"
        return "confirm_switch_mid_test"

    # ---------- test ----------

    # Picks the test items from the lesson and enters test mode.
    def build_test(self, state: TutorState) -> dict[str, Any]:
        items = build_test_items(state["lesson"], n=self.test_size, seed=self.seed)
        message = self._say(state.get("profile") or {}, "test_start", n=len(items))
        return {"mode": "test", "test_items": items, "test_index": 0, "test_results": [],
                "current_answer": new_answer(), "review_items": [], "intent": "",
                "messages": [AIMessage(message)]}

    # Waits for the learner's canvas answer (already uploaded by the UI) for the current item.
    def present_item(self, state: TutorState) -> dict[str, Any]:
        answer = dict(state.get("current_answer") or new_answer())
        payload = interrupt({"kind": "canvas", **self._item_info(state)})
        if self._is_chat(payload):
            return self._hold_chat(payload, "present_item")
        if isinstance(payload, dict) and payload.get("type") == "canvas":
            answer["image_path"] = payload.get("image_path")
            answer["strokes_path"] = payload.get("strokes_path")
            answer["reveal_count"] = int(payload.get("reveal_count") or 1)
        return {"current_answer": answer}

    # Returns index, total, target and attempt number for the UI's interrupt payloads.
    def _item_info(self, state: TutorState) -> dict[str, Any]:
        index = state.get("test_index", 0)
        return {"index": index, "total": len(state["test_items"]), "target": state["test_items"][index],
                "attempt": (state.get("current_answer") or {}).get("attempt", 0)}

    # Chat goes to test_chat; otherwise re-asks for the canvas until a valid upload arrives.
    def route_after_present(self, state: TutorState) -> str:
        if state.get("test_message"):
            return "route_message"
        return "read_letters" if (state.get("current_answer") or {}).get("image_path") else "present_item"

    # True when a resume payload is a typed chat message.
    def _is_chat(self, payload: Any) -> bool:
        return isinstance(payload, dict) and payload.get("type") == "message" and bool(str(payload.get("text") or "").strip())

    # Keeps a chat message typed during the test, and the step to return to afterwards.
    def _hold_chat(self, payload: dict[str, Any], step: str) -> dict[str, Any]:
        text = str(payload["text"]).strip()
        return {"messages": [HumanMessage(text)], "test_message": text, "resume_step": step}

    # The examiner answers a message typed during the test, without giving answers away.
    def test_chat(self, state: TutorState) -> dict[str, Any]:
        prompt = build_test_chat_prompt(state.get("profile") or {}, state["lesson"])
        history = self._chat_history(state.get("messages") or [])
        reply = self.models.chat.invoke([SystemMessage(prompt)] + history)
        return {"test_message": "", "messages": [AIMessage(reply.text.strip())]}

    # After the examiner's reply, return to the test step the learner was on.
    def route_after_test_chat(self, state: TutorState) -> str:
        return state.get("resume_step") or "present_item"

    # Leaves the test without a score and without saving; a confirmed switch with one lesson is chosen here.
    def exit_test(self, state: TutorState) -> dict[str, Any]:
        offer = state.get("switch_offer") or {}
        options = offer.get("options") or []
        choice = options[0] if offer.get("confirmed") and len(options) == 1 else ""
        return {"mode": "learn", "test_items": [], "test_index": 0, "current_answer": {}, "test_results": [],
                "test_message": "", "resume_step": "", "intent": "", "switch_choice": choice}

    # After leaving the test: apply a chosen switch, ask which of two lessons, or keep studying.
    def route_after_exit(self, state: TutorState) -> str:
        offer = state.get("switch_offer") or {}
        if state.get("switch_choice"):
            return "apply_switch"
        if offer.get("options") and (offer.get("confirmed") or offer.get("pending")):
            return "offer_switch"
        return "teacher"

    # Downloads the answer PNG and reads its letters with the OCR.
    def read_letters(self, state: TutorState) -> dict[str, Any]:
        answer = dict(state["current_answer"])
        png_bytes = self.db.download_attempt_image(answer["image_path"])
        answer["ocr_text"] = self.models.read_handwriting(png_bytes)
        answer["confirmed"] = None
        message = self._say(state.get("profile") or {}, "confirm_read", text=answer["ocr_text"] or "—")
        return {"current_answer": answer, "messages": [AIMessage(message)]}

    # Waits for the learner to confirm or dispute what the OCR read.
    def confirm_read(self, state: TutorState) -> dict[str, Any]:
        answer = dict(state["current_answer"])
        payload = interrupt({"kind": "confirm_read", "ocr_text": answer.get("ocr_text"), **self._item_info(state)})
        if self._is_chat(payload):
            return self._hold_chat(payload, "confirm_read")
        if not (isinstance(payload, dict) and payload.get("type") == "confirm_read"):
            return {"current_answer": answer}
        if payload.get("ok") is True:
            answer["confirmed"] = True
        else:
            answer["confirmed"] = False
            answer["dispute_count"] = int(answer.get("dispute_count") or 0) + 1
            answer["ocr_disputed"] = True
        return {"current_answer": answer}

    # Confirmed reads go to checking; first dispute retries; second dispute skips.
    def route_after_confirm(self, state: TutorState) -> str:
        answer = state.get("current_answer") or {}
        if state.get("test_message"):
            return "route_message"
        if answer.get("confirmed") is None:
            return "confirm_read"
        if answer["confirmed"]:
            return "check_letters"
        return "skip_item" if answer.get("dispute_count", 0) >= 2 else "retry_item"

    # Logs the disputed read (not counted) and asks for the same item again on a fresh canvas.
    def retry_item(self, state: TutorState) -> dict[str, Any]:
        answer = state["current_answer"]
        self._record_uncounted(state, answer)
        fresh = new_answer(attempt=answer.get("attempt", 0) + 1, dispute_count=answer.get("dispute_count", 0))
        return {"current_answer": fresh, "messages": [AIMessage(self._say(state.get("profile") or {}, "retry"))]}

    # Skips the item after a second dispute: logged for review, excluded from the score.
    def skip_item(self, state: TutorState) -> dict[str, Any]:
        answer = state["current_answer"]
        self._record_uncounted(state, answer)
        result = {"target": self._target(state), "ocr_text": answer.get("ocr_text"), "letters_correct": None,
                  "harakat_correct": None, "is_correct": None, "counted": False, "errors": []}
        return {"test_results": (state.get("test_results") or []) + [result],
                "messages": [AIMessage(self._say(state.get("profile") or {}, "skipped"))]}

    # Saves a disputed attempt with counted=false so its image can be reviewed later.
    def _record_uncounted(self, state: TutorState, answer: dict[str, Any]) -> None:
        self.db.record_attempt(
            user_id=state["user_id"], lesson_id=state["lesson"]["lesson_id"], target=self._target(state),
            ocr_text=answer.get("ocr_text"), ocr_disputed=True, letters_correct=None, harakat_input=None,
            harakat_correct=None, is_correct=None, counted=False, errors=[],
            reveal_count=int(answer.get("reveal_count") or 1),
            image_path=answer.get("image_path"), strokes_path=answer.get("strokes_path"),
        )

    # Returns the target text of the current test item.
    def _target(self, state: TutorState) -> str:
        return state["test_items"][state.get("test_index", 0)]

    # Compares the confirmed OCR letters with the target's letters.
    def check_letters(self, state: TutorState) -> dict[str, Any]:
        answer = dict(state["current_answer"])
        result = compare_letters(self._target(state), answer.get("ocr_text") or "")
        answer["letters_correct"] = result["letters_correct"]
        answer["errors"] = result["errors"]
        return {"current_answer": answer}

    # Wrong letters or levels 0/4 go to grading; right letters on levels 1-3 go to the harakat taps.
    def route_after_check(self, state: TutorState) -> str:
        letters_correct = (state.get("current_answer") or {}).get("letters_correct")
        if letters_correct and harakat_tested(state["lesson"]):
            return "ask_harakat"
        return "grade_item"

    # Waits for the learner's tapped marks, one entry per letter.
    def ask_harakat(self, state: TutorState) -> dict[str, Any]:
        answer = dict(state["current_answer"])
        letters = [entry["letter"] for entry in split_letters_and_marks(self._target(state))]
        payload = interrupt({"kind": "harakat", "letters": letters,
                             "allowed": allowed_marks_for(state["lesson"]), **self._item_info(state)})
        if self._is_chat(payload):
            return self._hold_chat(payload, "ask_harakat")
        if isinstance(payload, dict) and payload.get("type") == "harakat" and isinstance(payload.get("marks"), list):
            answer["harakat_input"] = payload["marks"]
        return {"current_answer": answer}

    # Chat goes to test_chat; otherwise re-asks for the marks until a valid list arrives.
    def route_after_harakat(self, state: TutorState) -> str:
        if state.get("test_message"):
            return "route_message"
        return "grade_item" if (state.get("current_answer") or {}).get("harakat_input") is not None else "ask_harakat"

    # Grades the item, gives feedback, and records the attempt.
    def grade_item(self, state: TutorState) -> dict[str, Any]:
        answer = state["current_answer"]
        result = self._grade(state, answer)
        prompt = build_feedback_prompt(state.get("profile") or {}, state["lesson"], result)
        feedback = self.models.chat.invoke([SystemMessage(prompt), HumanMessage("Give the feedback now.")])
        self.db.record_attempt(
            user_id=state["user_id"], lesson_id=state["lesson"]["lesson_id"], target=result["target"],
            ocr_text=answer.get("ocr_text"), ocr_disputed=bool(answer.get("ocr_disputed")),
            letters_correct=result["letters_correct"], harakat_input=answer.get("harakat_input"),
            harakat_correct=result["harakat_correct"], is_correct=result["is_correct"], counted=True,
            errors=result["errors"], reveal_count=int(answer.get("reveal_count") or 1),
            image_path=answer.get("image_path"), strokes_path=answer.get("strokes_path"),
        )
        return {"test_results": (state.get("test_results") or []) + [result],
                "messages": [AIMessage(feedback.text.strip())]}

    # Combines the letter result and (when tested) the harakat result into one item result.
    def _grade(self, state: TutorState, answer: dict[str, Any]) -> dict[str, Any]:
        target = self._target(state)
        letters_correct = bool(answer.get("letters_correct"))
        errors = list(answer.get("errors") or [])
        harakat_correct = None
        if letters_correct and harakat_tested(state["lesson"]):
            harakat = compare_harakat(target, answer.get("harakat_input") or [])
            harakat_correct = harakat["harakat_correct"]
            errors += harakat["errors"]
        is_correct = letters_correct and harakat_correct is not False
        return {"target": target, "ocr_text": answer.get("ocr_text"), "letters_correct": letters_correct,
                "harakat_correct": harakat_correct, "is_correct": is_correct, "counted": True, "errors": errors}

    # Moves to the next test item with a fresh answer.
    def next_item(self, state: TutorState) -> dict[str, Any]:
        return {"test_index": state.get("test_index", 0) + 1, "current_answer": new_answer()}

    # Presents the next item, or finishes the test after the last one.
    def route_after_next(self, state: TutorState) -> str:
        return "present_item" if state.get("test_index", 0) < len(state.get("test_items") or []) else "finish_test"

    # Scores counted items, saves progress, and moves to the next lesson on a pass.
    def finish_test(self, state: TutorState) -> dict[str, Any]:
        profile = dict(state.get("profile") or {})
        lesson = state["lesson"]
        counted = [result for result in state.get("test_results") or [] if result["counted"]]
        if not counted:
            return {"mode": "learn", "messages": [AIMessage(self._say(profile, "no_score"))]}
        score = sum(1 for result in counted if result["is_correct"]) / len(counted)
        passed = score >= float(lesson.get("pass_threshold") or 0.8)
        self.db.save_test_result(state["user_id"], lesson["lesson_id"], score, passed)
        percent = round(score * 100)
        if not passed:
            review = [result["target"] for result in counted if not result["is_correct"]]
            return {"mode": "learn", "review_items": review, "messages": [AIMessage(self._say(profile, "failed", score=percent))]}
        next_lesson = self.db.get_next_lesson(lesson["lesson_id"])
        if next_lesson is None:
            return {"mode": "learn", "messages": [AIMessage(self._say(profile, "passed_last", score=percent))]}
        profile = self.db.upsert_profile(state["user_id"], current_lesson_id=next_lesson["lesson_id"])
        return {"mode": "learn", "profile": profile, "review_items": [],
                "messages": [AIMessage(self._say(profile, "passed", score=percent))]}
