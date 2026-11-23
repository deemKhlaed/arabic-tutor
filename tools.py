"""Pure helpers: canvas to PNG, Arabic text checks, test-set building, prompts and parsing. No I/O."""

import difflib
import io
import random
import re
import unicodedata
from typing import Any

import numpy as np
from PIL import Image, ImageOps

# ---------- Arabic characters ----------

TATWEEL = "ـ"
SHADDA = "ّ"

# Tanween, short vowels, shadda, sukun, maddah/hamza marks and dagger alif.
HARAKAT_PATTERN = re.compile("[ً-ٰٕ]")

VOWEL_BY_CHAR = {
    "َ": "fatha",
    "ِ": "kasra",
    "ُ": "damma",
    "ْ": "sukun",
    "ً": "fathatan",
    "ٌ": "dammatan",
    "ٍ": "kasratan",
}
CHAR_BY_VOWEL = {name: char for char, name in VOWEL_BY_CHAR.items()}

ALL_VOWELS = ["fatha", "kasra", "damma", "sukun", "fathatan", "dammatan", "kasratan", "none"]

# Marks each level teaches; marks present in the lesson items are added on top.
LEVEL_VOWELS = {
    1: ["fatha"],
    2: ["fatha", "kasra", "damma"],
    3: ["fatha", "kasra", "damma", "sukun", "fathatan", "dammatan", "kasratan"],
}


# Normalizes to NFC and removes tatweel, whitespace, punctuation and invisible format characters.
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if char == TATWEEL or char.isspace() or category.startswith("P") or category == "Cf":
            continue
        kept.append(char)
    return "".join(kept)


# Removes every haraka (tanween, short vowels, shadda, sukun, dagger alif) after NFC composition.
def strip_harakat(text: str) -> str:
    composed = unicodedata.normalize("NFC", text or "")
    return HARAKAT_PATTERN.sub("", composed)


# Keeps only Arabic letters, dropping any extra words or symbols a model adds around its answer.
def arabic_letters_only(text: str) -> str:
    cleaned = strip_harakat(normalize(text))
    return "".join(char for char in cleaned if "؀" <= char <= "ۿ" and unicodedata.category(char) == "Lo")


# Compares the letters of target and OCR text exactly and lists per-character differences.
def compare_letters(target: str, ocr_text: str) -> dict[str, Any]:
    expected = strip_harakat(normalize(target))
    got = strip_harakat(normalize(ocr_text))
    errors = []
    matcher = difflib.SequenceMatcher(a=expected, b=got, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            errors.extend(_letter_errors(expected, got, i1, i2, j1, j2))
    return {"letters_correct": expected == got, "errors": errors}


# Turns one difflib opcode span into letter errors, pairing characters position by position.
def _letter_errors(expected: str, got: str, i1: int, i2: int, j1: int, j2: int) -> list[dict[str, Any]]:
    errors = []
    span = max(i2 - i1, j2 - j1)
    for offset in range(span):
        expected_char = expected[i1 + offset] if i1 + offset < i2 else ""
        got_char = got[j1 + offset] if j1 + offset < j2 else ""
        position = min(i1 + offset, len(expected))
        errors.append({"kind": "letter", "position": position, "expected": expected_char, "got": got_char})
    return errors


# Splits text into base letters, each with its shadda flag and vowel name ("none" if unmarked).
def split_letters_and_marks(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for char in normalize(text):
        if char == SHADDA and entries:
            entries[-1]["shadda"] = True
        elif char in VOWEL_BY_CHAR and entries:
            entries[-1]["vowel"] = VOWEL_BY_CHAR[char]
        elif unicodedata.category(char) == "Lo":
            entries.append({"letter": char, "shadda": False, "vowel": "none"})
    return entries


# Describes one letter's marks as text, e.g. "shadda+fatha", "kasra" or "none".
def mark_label(shadda: bool, vowel: str) -> str:
    return f"shadda+{vowel}" if shadda else vowel


# Compares the learner's tapped marks with the target's marks, letter by letter.
def compare_harakat(target: str, marks: list[dict[str, Any]]) -> dict[str, Any]:
    expected_entries = split_letters_and_marks(target)
    errors = []
    for position, expected in enumerate(expected_entries):
        tapped = marks[position] if position < len(marks) else {}
        got_shadda = bool(tapped.get("shadda", False))
        got_vowel = tapped.get("vowel") or "none"
        if got_shadda != expected["shadda"] or got_vowel != expected["vowel"]:
            errors.append({
                "kind": "haraka",
                "position": position,
                "letter": expected["letter"],
                "expected": mark_label(expected["shadda"], expected["vowel"]),
                "got": mark_label(got_shadda, got_vowel),
            })
    return {"harakat_correct": not errors, "errors": errors}


# Builds the display form of a letter with the chosen marks, for the harakat tiles.
def letter_with_marks(letter: str, shadda: bool, vowel: str) -> str:
    return letter + (SHADDA if shadda else "") + CHAR_BY_VOWEL.get(vowel, "")


# ---------- lesson items ----------

# Lists every item of a lesson as {text, meaning_en}, handling all target_items shapes.
def lesson_items(lesson: dict[str, Any]) -> list[dict[str, Any]]:
    target = lesson.get("target_items")
    if target is None:
        items = _items_from_rules_and_exercises(lesson)
    elif isinstance(target, dict):
        items = _items_from_groups(target)
    else:
        items = [item for entry in target for item in _items_from_entry(entry)]
    return _unique_by_text(items)


# Extracts items from one target_items entry (letter, syllable, word, harakat set or tanween forms).
def _items_from_entry(entry: Any) -> list[dict[str, Any]]:
    if isinstance(entry, str):
        return [{"text": entry, "meaning_en": None}]
    if not isinstance(entry, dict):
        return []
    meaning = entry.get("meaning_en")
    if "word" in entry:
        return [{"text": entry["word"], "meaning_en": meaning}]
    if "syllable" in entry:
        return [{"text": entry["syllable"], "meaning_en": None}]
    if "base" in entry and isinstance(entry.get("forms"), dict):
        return [{"text": form, "meaning_en": meaning} for form in entry["forms"].values()]
    if all(key in entry for key in ("fatha", "kasra", "damma")):
        return [{"text": entry[key], "meaning_en": None} for key in ("fatha", "kasra", "damma")]
    if "letter" in entry:
        return [{"text": entry["letter"], "meaning_en": entry.get("name_en")}]
    return []


# Extracts items from grouped target_items such as L3.07's {taa_marbuta: [...], ...}.
def _items_from_groups(groups: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for group in groups.values():
        for entry in group or []:
            items.extend(_items_from_entry(entry))
    return items


# Level 4 lessons: items are rule examples plus dictation_word exercise items.
def _items_from_rules_and_exercises(lesson: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for rule in lesson.get("rules") or []:
        items.extend({"text": example, "meaning_en": None} for example in rule.get("examples") or [])
    for exercise in lesson.get("exercises") or []:
        if exercise.get("type") == "dictation_word":
            items.extend({"text": text, "meaning_en": None} for text in exercise.get("items") or [] if isinstance(text, str))
    return items


# Removes duplicate items while keeping their first-seen order.
def _unique_by_text(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for item in items:
        if item["text"] and item["text"] not in seen:
            seen.add(item["text"])
            unique.append(item)
    return unique


# Picks n test targets (or all, if fewer) from the lesson without replacement.
def build_test_items(lesson: dict[str, Any], n: int = 5, seed: int | None = None) -> list[str]:
    texts = [item["text"] for item in lesson_items(lesson)]
    return random.Random(seed).sample(texts, min(n, len(texts)))


# True when the lesson's level tests harakat with the tap step (levels 1-3).
def harakat_tested(lesson: dict[str, Any]) -> bool:
    return lesson.get("level") in (1, 2, 3)


# Lists the vowel buttons and whether a shadda toggle is offered for this lesson.
def allowed_marks_for(lesson: dict[str, Any]) -> dict[str, Any]:
    level_vowels = set(LEVEL_VOWELS.get(lesson.get("level"), []))
    present_vowels: set[str] = set()
    shadda_present = False
    for item in lesson_items(lesson):
        for entry in split_letters_and_marks(item["text"]):
            present_vowels.add(entry["vowel"])
            shadda_present = shadda_present or entry["shadda"]
    wanted = level_vowels | present_vowels | {"none"}
    vowels = [vowel for vowel in ALL_VOWELS if vowel in wanted]
    return {"vowels": vowels, "shadda": shadda_present or lesson.get("level") == 3}


# ---------- canvas ----------

# Flattens RGBA canvas pixels onto white, crops to the ink with a margin, and returns PNG bytes.
def canvas_to_png(image_data: np.ndarray, margin: int = 24) -> bytes:
    rgba = Image.fromarray(np.asarray(image_data, dtype=np.uint8)).convert("RGBA")
    flat = Image.new("RGB", rgba.size, (255, 255, 255))
    flat.paste(rgba, mask=rgba.getchannel("A"))
    box = _ink_bounding_box(flat)
    if box is not None:
        left, top, right, bottom = box
        flat = flat.crop((
            max(left - margin, 0),
            max(top - margin, 0),
            min(right + margin, flat.width),
            min(bottom + margin, flat.height),
        ))
    flat = _upscale_small(flat)
    buffer = io.BytesIO()
    flat.save(buffer, format="PNG")
    return buffer.getvalue()


# Enlarges small crops to at least 256 px tall so vision models see dots and hamzas clearly.
def _upscale_small(image: Image.Image, min_height: int = 256) -> Image.Image:
    if image.height >= min_height:
        return image
    scale = min_height / image.height
    return image.resize((round(image.width * scale), min_height), Image.Resampling.LANCZOS)


# Finds the box around non-white pixels, or None when nothing was drawn.
def _ink_bounding_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    ink = ImageOps.invert(image.convert("L")).point(lambda value: 255 if value > 20 else 0)
    return ink.getbbox()


# True when the canvas JSON has no drawn objects.
def is_canvas_empty(json_data: dict[str, Any] | None) -> bool:
    return not json_data or not json_data.get("objects")


# ---------- onboarding answers ----------

LANGUAGE_NAMES = {
    "ar": "Arabic", "en": "English", "ur": "Urdu", "id": "Indonesian", "ms": "Malay", "tr": "Turkish",
    "fr": "French", "es": "Spanish", "de": "German", "hi": "Hindi", "bn": "Bengali", "fa": "Persian",
    "sw": "Swahili", "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
    "it": "Italian", "so": "Somali", "ha": "Hausa", "tl": "Tagalog", "uz": "Uzbek", "ps": "Pashto",
    "nl": "Dutch", "pl": "Polish", "th": "Thai", "vi": "Vietnamese", "ta": "Tamil", "am": "Amharic",
}


# Returns a readable language name for an ISO code, falling back to the code itself.
def language_name(code: str | None) -> str:
    return LANGUAGE_NAMES.get((code or "").lower(), code or "English")


# Reads "child" or "non_native" from a free-text answer; None when unclear.
def parse_learner_type(text: str) -> str | None:
    answer = (text or "").strip().lower()
    non_native_words = ["non", "not native", "learner", "adult", "foreign", "2", "غير", "أجنبي", "اجنبي"]
    child_words = ["child", "kid", "طفل", "ابن", "بنت", "1"]
    if any(word in answer for word in non_native_words):
        return "non_native"
    if any(word in answer for word in child_words):
        return "child"
    return None


# Reads a start level 0-4 from text; "I don't know" means 0; None when unclear.
def parse_start_level(text: str) -> int | None:
    answer = (text or "").strip().lower().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    unsure_words = ["don't know", "dont know", "do not know", "not sure", "idk", "لا أعرف", "لا اعرف", "لا أدري"]
    if any(word in answer for word in unsure_words):
        return 0
    match = re.search(r"\d+", answer)
    if match and 0 <= int(match.group()) <= 4:
        return int(match.group())
    return None


# Reads an ISO language code from a code or English language name; None when unknown.
def parse_language_code(text: str) -> str | None:
    answer = (text or "").strip().lower()
    if answer in LANGUAGE_NAMES:
        return answer
    for code, name in LANGUAGE_NAMES.items():
        if name.lower() in answer:
            return code
    return None


# True when a string looks like a two-letter ISO 639-1 code.
def is_language_code(text: str) -> bool:
    return bool(re.fullmatch(r"[a-z]{2}", (text or "").strip().lower()))


# ---------- lesson routing (RAG over lesson cards) ----------

# Routing thresholds, tuned with data/routing_eval.jsonl: fast-path "stay" needs this similarity and lead.
MIN_SIM = 0.45
MARGIN = 0.05
ROUTER_TIMEOUT_SECONDS = 8
ROUTE_INTENTS = ("stay", "switch", "ready", "chat", "stop")

ROUTER_PROMPT = (
    "You route one message from a learner in an Arabic spelling course. The learner may write Saudi dialect, "
    "standard Arabic or English, with typos. Choose exactly one intent:\n"
    "- stay: a question or request about the CURRENT lesson's topic.\n"
    "- switch: they want to learn a different topic or lesson (one of the candidates), or something harder, "
    "easier or next.\n"
    "- ready: they say they are ready to be tested or want the test now.\n"
    "- stop: during a test, they want to stop the test and go back to studying the current lesson.\n"
    "- chat: greetings, thanks, small talk or anything off-topic.\n"
    "For switch, lesson_id must be one of the candidate ids; for 'harder' or 'next' use the candidate marked "
    "NEXT. Set unsure to true when the message could fit two candidates or names no clear topic. "
    "For other intents lesson_id is null.\n"
    'Answer with JSON only: {"intent": "stay|switch|ready|chat|stop", "lesson_id": "<candidate id or null>", '
    '"unsure": true|false}'
)


# Returns a lesson's items, or an empty list if its content has an unexpected shape.
def safe_lesson_items(lesson: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return lesson_items(lesson)
    except Exception:
        return []


# Lists the base letters and the mark names that a lesson's items use, in first-seen order.
def letters_and_marks(lesson: dict[str, Any]) -> tuple[list[str], list[str]]:
    letters: list[str] = []
    marks: list[str] = []
    for item in safe_lesson_items(lesson):
        for entry in split_letters_and_marks(item["text"]):
            if entry["letter"] not in letters:
                letters.append(entry["letter"])
            label = mark_label(entry["shadda"], entry["vowel"])
            if label != "none" and label not in marks:
                marks.append(label)
    return letters, marks


# Builds the text embedded for one lesson: titles, objective, skill, letters, examples, rules and sample questions.
def build_card_text(lesson: dict[str, Any], sample_questions: list[str]) -> str:
    letters, marks = letters_and_marks(lesson)
    parts = [lesson.get("title_ar") or "", lesson.get("title_en") or "", lesson.get("objective_en") or "",
             f"skill: {lesson.get('skill') or ''}"]
    if letters:
        parts.append("letters: " + " ".join(letters[:30]))
    if marks:
        parts.append("marks: " + ", ".join(marks))
    examples = [item["text"] for item in safe_lesson_items(lesson)[:8]]
    if examples:
        parts.append("examples: " + " ".join(examples))
    if int(lesson.get("level") or 0) == 4:
        parts.extend(rule.get("condition_ar") or "" for rule in lesson.get("rules") or [])
    parts.extend(sample_questions)
    return "\n".join(part for part in parts if part)


# One candidate line for the router: id, titles, objective, similarity, and a NEXT / CURRENT marker.
def candidate_line(lesson: dict[str, Any], similarity: float | None, marker: str = "") -> str:
    score = f" | similarity {similarity:.2f}" if similarity is not None else ""
    tag = f" [{marker}]" if marker else ""
    return (f"{lesson['lesson_id']}{tag} | {lesson.get('title_ar') or ''} / {lesson.get('title_en') or ''} | "
            f"{lesson.get('objective_en') or ''}{score}")


# Builds the router's input: mode, current lesson, candidates with scores, and the learner's message.
def router_request(message: str, mode: str, current: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    lines = [f"Mode: {mode}", "Current lesson: " + candidate_line(current, None), "Candidates:"]
    for candidate in candidates:
        lines.append("- " + candidate_line(candidate["lesson"], candidate.get("similarity"), candidate.get("marker", "")))
    lines.append(f"Learner message: {message}")
    return "\n".join(lines)


# Reads the router's JSON defensively; garbage, unknown intents or non-candidate lessons become "stay".
def parse_route(text: str, candidate_ids: list[str]) -> dict[str, Any]:
    fallback = {"intent": "stay", "lesson_id": None, "unsure": False, "valid": False}
    data = parse_json_object(text or "")
    intent = str(data.get("intent") or "").strip().lower()
    if intent not in ROUTE_INTENTS:
        return fallback
    lesson_id = data.get("lesson_id")
    lesson_id = str(lesson_id).strip() if lesson_id not in (None, "", "null") else None
    unsure = data.get("unsure") is True or str(data.get("unsure")).lower() == "true"
    if intent != "switch":
        return {"intent": intent, "lesson_id": None, "unsure": False, "valid": True}
    if lesson_id is not None and lesson_id not in candidate_ids:
        return fallback
    if lesson_id is None and not unsure:
        return fallback
    return {"intent": "switch", "lesson_id": lesson_id, "unsure": unsure, "valid": True}


# True when retrieval alone says "stay": the current lesson leads clearly with enough similarity.
def is_fast_stay(matches: list[dict[str, Any]], current_id: str) -> bool:
    if not matches or matches[0]["lesson_id"] != current_id or matches[0]["similarity"] < MIN_SIM:
        return False
    second = matches[1]["similarity"] if len(matches) > 1 else 0.0
    return matches[0]["similarity"] - second >= MARGIN


# Picks the lessons to offer: the router's choice, plus the runner-up when unsure or retrieval is too close.
def switch_options(route: dict[str, Any], matches: list[dict[str, Any]], current_id: str) -> list[str]:
    others = [match for match in matches if match["lesson_id"] != current_id]
    chosen = route.get("lesson_id")
    close = len(others) >= 2 and others[0]["similarity"] - others[1]["similarity"] < MARGIN
    top_two = [match["lesson_id"] for match in others[:2]]
    if chosen is None:
        return top_two if route.get("unsure") else []
    if chosen == current_id:
        return []
    if route.get("unsure") or (close and chosen in top_two):
        runner_up = next((lesson_id for lesson_id in top_two if lesson_id != chosen), None)
        return [chosen, runner_up] if runner_up else [chosen]
    return [chosen]


# ---------- prompts ----------


# Builds the examiner's prompt for answering a question during the test without giving answers away.
def build_test_chat_prompt(profile: dict[str, Any], lesson: dict[str, Any]) -> str:
    return "\n".join([
        f"You are the examiner during a spelling test on the lesson: {lesson.get('title_en')}.",
        _style_prompt(profile),
        "Answer the learner's last message briefly (at most 40 words). Never write, spell or hint at the "
        "letters or harakat of any test item. Help only with how to use the test (writing on the canvas, "
        "'Show again', the read check, tapping marks). If they greet you or say they are ready, reply in one "
        "short sentence and let them continue the test.",
    ])


# Builds the teacher's system prompt from the lesson and the learner profile.
def build_system_prompt(profile: dict[str, Any], lesson: dict[str, Any], review_items: list[str] | None = None) -> str:
    parts = [_lesson_prompt(lesson), _style_prompt(profile)]
    if review_items:
        parts.append(_review_prompt(review_items))
    parts.append(_readiness_prompt(profile))
    return "\n\n".join(parts)


# Describes the lesson: title, objective, writing guide, items and rules.
def _lesson_prompt(lesson: dict[str, Any]) -> str:
    lines = [
        "You are a patient Arabic spelling (إملاء) teacher.",
        f"Lesson {lesson.get('lesson_id')}: {lesson.get('title_en')} / {lesson.get('title_ar')}",
        f"Objective: {lesson.get('objective_en')}",
    ]
    if lesson.get("writing_guide_en"):
        lines.append(f"How it is written: {lesson['writing_guide_en']}")
    lines.append("Lesson items (use these as your examples):")
    for item in lesson_items(lesson)[:40]:
        meaning = f" ({item['meaning_en']})" if item.get("meaning_en") else ""
        lines.append(f"- {item['text']}{meaning}")
    lines.extend(_rules_lines(lesson))
    lines.append(
        "How to teach: by default teach this lesson's concept, based on the objective and writing guide above. "
        "Stay on this lesson's concept. If the learner's message is small talk or off-topic, reply in one short "
        "sentence and steer back to the lesson; the app itself handles requests to change lesson. "
        "If the learner asked something, answer that first, briefly. "
        "Go in small steps: one point and one or two example words per reply. "
        "For this lesson, take example words from the lesson items above, copied exactly with their harakat. "
        "Do not quiz the learner yourself; the app runs the test."
    )
    return "\n".join(lines)


# Lists the lesson's spelling rules with their examples, if any.
def _rules_lines(lesson: dict[str, Any]) -> list[str]:
    rules = lesson.get("rules") or []
    if not rules:
        return []
    lines = ["Spelling rules of this lesson (do not invent other rules):"]
    for rule in rules:
        examples = "، ".join(rule.get("examples") or [])
        spelling = f" → {rule['spelling']}" if rule.get("spelling") else ""
        lines.append(f"- [{rule['rule_id']}] {rule.get('condition_en')}{spelling}. Examples: {examples}")
    return lines


# Gives the tone and language rules for a child or a non-native learner.
def _style_prompt(profile: dict[str, Any]) -> str:
    if profile.get("learner_type") == "child":
        return (
            "The learner is an Arabic-speaking child. Reply only in simple Modern Standard Arabic with short sentences. "
            "One idea per message. Repeat the key point. Be very patient; if the child does not understand, "
            "explain again in a different way. Praise effort and never shame mistakes. At most about 60 words per reply."
        )
    language = language_name(profile.get("explain_language"))
    return (
        f"The learner is a non-native speaker. Explain in {language}. "
        f"For every Arabic word, add a Latin transliteration and its meaning in {language}. "
        f"Compare Arabic sounds to sounds in {language} where helpful. One idea per reply, at most about 100 words, "
        "no headings."
    )


# Asks the teacher to review the items the learner got wrong in the last test.
def _review_prompt(review_items: list[str]) -> str:
    return (
        "The learner just failed a test on this lesson. Gently review these items they got wrong: "
        + "، ".join(review_items)
    )


# Requires every reply to end with the readiness question in the learner's language.
def _readiness_prompt(profile: dict[str, Any]) -> str:
    language = "Arabic" if profile.get("learner_type") == "child" else language_name(profile.get("explain_language"))
    return (
        f"End every reply with one short question in {language} asking whether the learner "
        "is ready to test what they learned today."
    )


# Builds the examiner's system prompt for feedback on one graded item.
def build_feedback_prompt(profile: dict[str, Any], lesson: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [
        "You are the examiner in an Arabic spelling test. Give short feedback (at most 40 words) on one answer.",
        _style_prompt(profile),
        f"Lesson: {lesson.get('title_en')}. Target: {result['target']}. Learner's letters as read: {result.get('ocr_text')}.",
        f"Correct: {result['is_correct']}. Errors (positions start at 0): {result['errors']}",
        "Explain each error simply. If correct, praise briefly. Do not ask whether they are ready.",
    ]
    if lesson.get("level") == 4 and lesson.get("rules"):
        lines.extend(_rules_lines(lesson))
        lines.append("Cite the matching rule above by quoting it. Never invent a rule.")
    return "\n".join(lines)


# ---------- fixed messages ----------

MESSAGES = {
    "ar": {
        "switched": "لننتقل إلى الدرس {lesson_id}: {title}.",
        "switch_prereq": "هذا الدرس مبني على: {titles}، ولم تجتز اختباره بعد، لكن يمكننا البدء.",
        "offer_switch_one": "تبي ننتقل لدرس {title}؟",
        "offer_switch_two": "أي درس تقصد: {first} أم {second}؟",
        "switch_declined": "تمام، نبقى في درس {title}. وش تحب نسوي؟",
        "offer_test_switch": "طلبت درس {title}. تكمل الاختبار أولًا، أو نوقفه وننتقل الآن؟",
        "finish_first": "ممتاز، كمّل الاختبار، وبعده أعرض عليك الانتقال.",
        "test_start": "لِنَبْدَأِ الاختبار! سأريك {n} كلمات، واحدةً بعد الأخرى. اكتب كل واحدة على اللوح.",
        "confirm_read": "قرأتُ: {text} — هل هذا ما كتبتَ؟",
        "retry": "لا بأس، هذا خطأ في قراءتي وليس خطأك. اكتبها مرة أخرى من فضلك.",
        "skipped": "سنتجاوز هذه الكلمة، ولن تُحسب عليك.",
        "no_score": "لم أستطع قراءة أي إجابة، لذلك لا توجد درجة هذه المرة. سنبقى في الدرس نفسه.",
        "passed": "أحسنت! نتيجتك {score}٪. ننتقل الآن إلى الدرس التالي.",
        "passed_last": "أحسنت! نتيجتك {score}٪. لقد أنهيت كل الدروس!",
        "failed": "نتيجتك {score}٪. لنراجع الدرس معًا ثم نحاول مرة أخرى.",
        "ready_question": "هل أنت مستعد لاختبار ما تعلمته اليوم؟",
    },
    "en": {
        "switched": "Let's move to lesson {lesson_id}: {title}.",
        "switch_prereq": "This lesson builds on: {titles}, which you haven't passed yet, but we can start anyway.",
        "offer_switch_one": "Shall we move to the lesson {title}?",
        "offer_switch_two": "Which lesson do you mean: {first} or {second}?",
        "switch_declined": "OK, we'll stay with {title}. What would you like to do?",
        "offer_test_switch": "You asked for {title}. Finish this test first, or stop it and switch now?",
        "finish_first": "Great, finish the test, and I'll offer the switch afterwards.",
        "test_start": "Let's start the test! You will see {n} items, one at a time. Write each one on the canvas.",
        "confirm_read": "I read: {text} — is that what you wrote?",
        "retry": "No problem — that was my reading mistake, not yours. Please write it again.",
        "skipped": "We'll skip this one. It won't count against you.",
        "no_score": "I couldn't read any answer, so there is no score this time. We'll stay on this lesson.",
        "passed": "Well done! You scored {score}%. On to the next lesson.",
        "passed_last": "Well done! You scored {score}%. You have finished every lesson!",
        "failed": "You scored {score}%. Let's review the lesson together and try again.",
        "ready_question": "Are you ready to test what you learned today?",
    },
}


# The learner's interface language: Arabic for children, otherwise their explanation language.
def learner_language(profile: dict[str, Any]) -> str:
    if profile.get("learner_type") == "child":
        return "ar"
    return (profile.get("explain_language") or "en").lower()


# Picks the fixed-message table: Arabic or English; other languages are translated by Nodes.
def message_language(profile: dict[str, Any]) -> str:
    return "ar" if learner_language(profile) == "ar" else "en"


# Returns a fixed message in Arabic or English, filled with the given values.
def say(profile: dict[str, Any], key: str, **values: Any) -> str:
    return MESSAGES[message_language(profile)][key].format(**values)


RTL_LANGUAGES = {"ar", "ur", "fa", "ps", "he", "ug", "sd", "ku"}


# True when a language is written right to left.
def is_rtl(language_code: str) -> bool:
    return language_code in RTL_LANGUAGES


# ---------- interface text ----------

UI_TEXT = {
    "en": {
        "title": "Arabic spelling tutor", "account": "Account", "open_account": "Open **Account** to sign in or create an account.",
        "session_expired": "Your session expired. Please sign in again.",
        "interface_language": "Interface language", "sign_in": "Sign in", "sign_up": "Sign up", "email": "Email",
        "password": "Password", "name": "Name", "create_account": "Create account",
        "check_email": "Check your email to confirm the account, then sign in.",
        "sign_in_failed": "Sign-in failed", "sign_up_failed": "Sign-up failed", "learner_type": "Learner type",
        "child": "Child", "non_native": "Non-native speaker", "explain_language": "Explanation language (ISO code)",
        "reveal_seconds": "Seconds the word is shown", "save": "Save settings", "sign_out": "Sign out",
        "saved_later": "Saved. Teaching-style changes apply after the current step.",
        "next": "→ Next", "item_of": "Item {k} / {n}", "show_again": "Show again", "pen_width": "Pen width",
        "clear": "Clear", "submit": "Submit", "empty_canvas": "The canvas is empty. Write your answer first.",
        "pen_hint": "If the pen stops drawing, the ✎ button above the canvas is on (select mode): click it again, or press Clear.",
        "i_read": "I read:", "yes_wrote": "Yes, that's what I wrote", "no_wrote": "No, that's not what I wrote",
        "tap_marks": "Tap the mark on each letter:", "check": "Check", "shadda": "shadda",
        "yes": "Yes", "no": "No", "neither": "Neither", "finish_test_first": "Finish the test first",
        "stop_and_switch": "Stop and switch",
        "write_message": "Write a message…", "ready": "I'm ready", "retry": "Retry", "error": "Something went wrong",
        "gloss_unavailable": "(meaning unavailable)", "fatha": "fatha", "kasra": "kasra", "damma": "damma",
        "sukun": "sukun", "fathatan": "fathatan", "dammatan": "dammatan", "kasratan": "kasratan", "none": "no mark",
    },
    "ar": {
        "title": "مُعَلِّمُ الإِمْلاءِ", "account": "الحساب", "open_account": "افتح **الحساب** لتسجيل الدخول أو إنشاء حساب.",
        "session_expired": "انتهت جلستك. سجّل الدخول مرة أخرى.",
        "interface_language": "لغة الواجهة", "sign_in": "تسجيل الدخول", "sign_up": "حساب جديد", "email": "البريد الإلكتروني",
        "password": "كلمة المرور", "name": "الاسم", "create_account": "أنشئ الحساب",
        "check_email": "افتح بريدك لتأكيد الحساب، ثم سجّل الدخول.",
        "sign_in_failed": "تعذّر تسجيل الدخول", "sign_up_failed": "تعذّر إنشاء الحساب", "learner_type": "نوع المتعلّم",
        "child": "طفل", "non_native": "متعلّم غير ناطق بالعربية", "explain_language": "لغة الشرح (رمز اللغة)",
        "reveal_seconds": "ثواني ظهور الكلمة", "save": "احفظ الإعدادات", "sign_out": "تسجيل الخروج",
        "saved_later": "تمّ الحفظ. تُطبَّق التغييرات بعد الخطوة الحالية.",
        "next": "التالي ←", "item_of": "السؤال {k} من {n}", "show_again": "أرني مرة أخرى", "pen_width": "سُمك القلم",
        "clear": "امسح", "submit": "أرسل", "empty_canvas": "اللوح فارغ. اكتب إجابتك أولًا.",
        "pen_hint": "إذا توقّف القلم عن الكتابة فزرّ ✎ فوق اللوح مُفعَّل: اضغطه مرة أخرى أو اضغط «امسح».",
        "i_read": "قرأتُ:", "yes_wrote": "نعم، هذا ما كتبتُه", "no_wrote": "لا، لم أكتب هذا",
        "tap_marks": "اختر الحركة على كل حرف:", "check": "تحقّق", "shadda": "شدّة",
        "yes": "نعم", "no": "لا", "neither": "ولا واحد", "finish_test_first": "أكمل الاختبار أولًا",
        "stop_and_switch": "أوقف وانتقل",
        "write_message": "اكتب رسالة…", "ready": "أنا مستعد", "retry": "أعد المحاولة", "error": "حدث خطأ",
        "gloss_unavailable": "(المعنى غير متاح)", "fatha": "فتحة", "kasra": "كسرة", "damma": "ضمّة",
        "sukun": "سكون", "fathatan": "تنوين فتح", "dammatan": "تنوين ضمّ", "kasratan": "تنوين كسر", "none": "بلا حركة",
    },
}


# Reads a JSON object from a model reply, ignoring code fences; returns {} when it cannot.
def parse_json_object(text: str) -> dict[str, Any]:
    import json

    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group())
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


# Keeps a translated table only where it has every key; missing keys fall back to English.
def merge_translation(english: dict[str, str], translated: dict[str, Any]) -> dict[str, str]:
    merged = {}
    for key, text in english.items():
        candidate = translated.get(key)
        same_slots = isinstance(candidate, str) and sorted(re.findall(r"\{\w+\}", candidate)) == sorted(re.findall(r"\{\w+\}", text))
        merged[key] = candidate if same_slots else text
    return merged


# True when a reply already ends with a question mark (Latin or Arabic).
def ends_with_question(text: str) -> bool:
    return (text or "").rstrip().endswith(("?", "؟"))
