"""Admin script: checks the sample questions, builds one card per lesson, embeds it and upserts it into lesson_cards.

Run from arabic_tutor/:
    python build_lesson_cards.py --check   # only report ambiguous sample questions
    python build_lesson_cards.py           # check, then embed and upsert all cards (refuses if any are ambiguous)

It reads .env.admin (SUPABASE_URL, SUPABASE_SECRET_KEY, CF_ACCOUNT_ID, CF_AI_API_TOKEN). The app never loads it.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
CARDS_FILE = HERE / "data" / "lesson_cards.json"
CHECK_MARGIN = 0.01   # ambiguous: its own lesson beats the best other lesson by less than this (a tie or worse)


# Loads only .env.admin (next to this script or one folder up); stops if it is missing.
def load_admin_env() -> None:
    for folder in (HERE, HERE.parent):
        path = folder / ".env.admin"
        if path.exists():
            load_dotenv(path, override=True)
            return
    sys.exit("Missing .env.admin with SUPABASE_URL, SUPABASE_SECRET_KEY, CF_ACCOUNT_ID and CF_AI_API_TOKEN")


# Reads data/lesson_cards.json and checks it has one entry per course lesson.
def read_cards(course: list[dict[str, Any]]) -> dict[str, list[str]]:
    entries = json.loads(CARDS_FILE.read_text(encoding="utf-8"))
    cards = {entry["lesson_id"]: list(entry["sample_questions"]) for entry in entries}
    missing = [lesson["lesson_id"] for lesson in course if lesson["lesson_id"] not in cards]
    unknown = [lesson_id for lesson_id in cards if lesson_id not in {lesson["lesson_id"] for lesson in course}]
    if missing or unknown:
        sys.exit(f"lesson_cards.json does not match the course. Missing: {missing}. Unknown: {unknown}.")
    return cards


# Turns a list of vectors into unit-length rows so a dot product is the cosine similarity.
def unit_rows(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.array(vectors, dtype=float)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


# Scores every question against its own card (without that question) and every other card.
def ambiguity_report(course: list[dict[str, Any]], cards: dict[str, list[str]], models: Any) -> list[dict[str, Any]]:
    from tools import build_card_text

    lesson_ids = [lesson["lesson_id"] for lesson in course]
    full_cards = unit_rows(models.embed([build_card_text(lesson, cards[lesson["lesson_id"]]) for lesson in course]))
    pairs = [(lesson, question) for lesson in course for question in cards[lesson["lesson_id"]]]
    questions = unit_rows(models.embed([question for _, question in pairs]))
    left_out = [build_card_text(lesson, [q for q in cards[lesson["lesson_id"]] if q != question]) for lesson, question in pairs]
    own_cards = unit_rows(models.embed(left_out))
    report = []
    for row, (lesson, question) in enumerate(pairs):
        own_index = lesson_ids.index(lesson["lesson_id"])
        scores = full_cards @ questions[row]
        scores[own_index] = float(own_cards[row] @ questions[row])
        others = [(scores[i], lesson_ids[i]) for i in range(len(lesson_ids)) if i != own_index]
        best_other, best_other_id = max(others)
        report.append({"lesson_id": lesson["lesson_id"], "question": question, "own": float(scores[own_index]),
                       "best_other": float(best_other), "best_other_id": best_other_id,
                       "margin": float(scores[own_index] - best_other)})
    return report


# Prints the questions whose margin is below CHECK_MARGIN and returns them.
def print_ambiguous(report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flagged = sorted((row for row in report if row["margin"] < CHECK_MARGIN), key=lambda row: row["margin"])
    margins = np.array([row["margin"] for row in report])
    print(f"{len(report)} questions | margin median {np.median(margins):.3f} | min {margins.min():.3f} | "
          f"flagged (< {CHECK_MARGIN}) {len(flagged)}")
    for row in flagged:
        print(f"  {row['lesson_id']} vs {row['best_other_id']}  margin {row['margin']:+.3f}  "
              f"(own {row['own']:.3f}, other {row['best_other']:.3f})  {row['question']}")
    return flagged


# Embeds every card and upserts it into lesson_cards; re-running updates the rows in place.
def build_and_upsert(course: list[dict[str, Any]], cards: dict[str, list[str]], models: Any, db: Any) -> None:
    from tools import build_card_text

    texts = [build_card_text(lesson, cards[lesson["lesson_id"]]) for lesson in course]
    vectors = models.embed(texts)
    if any(len(vector) != 1024 for vector in vectors):
        sys.exit("Embedding length is not 1024; lesson_cards.embedding is vector(1024).")
    now = datetime.now(timezone.utc).isoformat()
    rows = [{"lesson_id": lesson["lesson_id"], "card_text": text, "sample_questions": cards[lesson["lesson_id"]],
             "embedding": vector, "updated_at": now}
            for lesson, text, vector in zip(course, texts, vectors)]
    db.upsert_lesson_cards(rows)
    print(f"Upserted {len(rows)} lesson cards.")


# Runs the check, and unless --check was given, builds and upserts the cards.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only report ambiguous sample questions")
    parser.add_argument("--allow-ambiguous", action="store_true", help="upsert even if some questions are flagged")
    args = parser.parse_args()
    load_admin_env()
    from Database import AdminDatabase
    from model import Models

    db, models = AdminDatabase(), Models()
    course = db.get_course()
    cards = read_cards(course)
    flagged = print_ambiguous(ambiguity_report(course, cards, models))
    if args.check:
        return
    if flagged and not args.allow_ambiguous:
        sys.exit("Fix the flagged questions first (or pass --allow-ambiguous).")
    build_and_upsert(course, cards, models, db)


if __name__ == "__main__":
    main()
