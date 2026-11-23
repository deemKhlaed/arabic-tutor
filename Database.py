"""Every Supabase call the app makes: auth, profiles, lessons, rules, progress, attempts and storage."""

import json
import os
import uuid
from typing import Any

from supabase import Client, create_client

ATTEMPTS_BUCKET = "attempts"


# Reads the public (anon / publishable) key; the service_role key is never used.
def read_public_key() -> str:
    key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    if not key:
        raise RuntimeError("Set SUPABASE_ANON_KEY (or SUPABASE_PUBLISHABLE_KEY) in .env")
    return key


class Database:
    # Creates one Supabase client (public key by default); after sign-in it carries the user's session so RLS applies.
    def __init__(self, api_key: str | None = None) -> None:
        url = os.environ.get("SUPABASE_URL")
        if not url:
            raise RuntimeError("Set SUPABASE_URL in .env")
        self.client: Client = create_client(url, api_key or read_public_key())
        self._course: list[dict[str, Any]] | None = None

    # ---------- auth ----------

    # Refreshes the access token if it expired (it lasts 1 hour); returns False if the user must sign in again.
    def keep_session_fresh(self) -> bool:
        try:
            return self.client.auth.get_session() is not None
        except Exception:
            return False

    # Returns a query builder for a table, refreshing an expired token first so RLS calls never fail.
    def _table(self, name: str) -> Any:
        self.keep_session_fresh()
        return self.client.table(name)

    # Returns the attempts storage bucket, refreshing an expired token first.
    def _bucket(self) -> Any:
        self.keep_session_fresh()
        return self.client.storage.from_(ATTEMPTS_BUCKET)

    # Creates an account; returns the user id, or None when email confirmation is still pending.
    def sign_up(self, email: str, password: str, display_name: str) -> str | None:
        response = self.client.auth.sign_up(
            {"email": email, "password": password, "options": {"data": {"display_name": display_name}}}
        )
        if response.user is None:
            raise RuntimeError("Sign-up failed")
        if response.session is None:
            return None
        self.upsert_profile(response.user.id, display_name=display_name)
        return response.user.id

    # Signs in with email and password and returns the user id.
    def sign_in(self, email: str, password: str) -> str:
        response = self.client.auth.sign_in_with_password({"email": email, "password": password})
        if response.user is None:
            raise RuntimeError("Sign-in failed")
        return response.user.id

    # Ends the current session.
    def sign_out(self) -> None:
        self.client.auth.sign_out()

    # Returns the signed-in user as {id, email, display_name}, or None.
    def get_user(self) -> dict[str, Any] | None:
        try:
            response = self.client.auth.get_user()
        except Exception:
            return None
        if response is None or response.user is None:
            return None
        metadata = response.user.user_metadata or {}
        return {
            "id": response.user.id,
            "email": response.user.email,
            "display_name": metadata.get("display_name"),
        }

    # ---------- profiles ----------

    # Returns the learner's profile row, or None if it does not exist yet.
    def get_profile(self, user_id: str) -> dict[str, Any] | None:
        rows = self._table("profiles").select("*").eq("user_id", user_id).limit(1).execute().data
        return rows[0] if rows else None

    # Inserts or updates the profile with the given fields and returns the saved row.
    def upsert_profile(self, user_id: str, **fields: Any) -> dict[str, Any]:
        row = {"user_id": user_id, **fields}
        saved = self._table("profiles").upsert(row, on_conflict="user_id").execute().data
        return saved[0] if saved else row

    # ---------- curriculum (read-only) ----------

    # Returns all levels ordered by number, used to offer start levels.
    def get_levels(self) -> list[dict[str, Any]]:
        return self._table("levels").select("level,title_en,title_ar").order("level").execute().data

    # Returns one lesson with its rules (ordered by position) under the key "rules".
    def get_lesson(self, lesson_id: str) -> dict[str, Any] | None:
        rows = self._table("lessons").select("*").eq("lesson_id", lesson_id).limit(1).execute().data
        if not rows:
            return None
        lesson = rows[0]
        lesson["rules"] = (
            self._table("rules").select("*").eq("lesson_id", lesson_id).order("position").execute().data
        )
        return lesson

    # Returns the first lesson (lowest position) of a level, with rules.
    def get_first_lesson_of_level(self, level: int) -> dict[str, Any] | None:
        rows = (
            self._table("lessons").select("lesson_id")
            .eq("level", level).order("position").limit(1).execute().data
        )
        return self.get_lesson(rows[0]["lesson_id"]) if rows else None

    # Returns every lesson with its rules, in course order; read once per session for course search.
    def get_course(self) -> list[dict[str, Any]]:
        if self._course is None:
            lessons = self._table("lessons").select("*").order("level").order("position").execute().data
            rules = self._table("rules").select("*").order("position").execute().data
            for lesson in lessons:
                lesson["rules"] = [rule for rule in rules if rule["lesson_id"] == lesson["lesson_id"]]
            self._course = lessons
        return self._course

    # Returns the k lessons whose cards are closest to a query embedding, as {lesson_id, similarity}.
    def match_lessons(self, embedding: list[float], k: int = 3) -> list[dict[str, Any]]:
        self.keep_session_fresh()
        rows = self.client.rpc("match_lessons", {"query_embedding": embedding, "match_count": k}).execute().data
        return [{"lesson_id": row["lesson_id"], "similarity": float(row["similarity"])} for row in rows or []]

    # Saves the learner's current lesson in their profile.
    def set_current_lesson(self, user_id: str, lesson_id: str) -> None:
        self.upsert_profile(user_id, current_lesson_id=lesson_id)

    # Returns the lesson after this one by (level, position), or None after the last lesson.
    def get_next_lesson(self, lesson_id: str) -> dict[str, Any] | None:
        ordered = (
            self._table("lessons").select("lesson_id,level,position")
            .order("level").order("position").execute().data
        )
        ids = [row["lesson_id"] for row in ordered]
        if lesson_id not in ids:
            return None
        next_index = ids.index(lesson_id) + 1
        return self.get_lesson(ids[next_index]) if next_index < len(ids) else None

    # ---------- progress ----------

    # Returns every lesson_progress row of the learner.
    def get_progress(self, user_id: str) -> list[dict[str, Any]]:
        return self._table("lesson_progress").select("*").eq("user_id", user_id).execute().data

    # Upserts progress: bumps tests_taken, keeps the best score, and keeps 'passed' once reached.
    def save_test_result(self, user_id: str, lesson_id: str, score: float, passed: bool) -> dict[str, Any]:
        existing = (
            self._table("lesson_progress").select("*")
            .eq("user_id", user_id).eq("lesson_id", lesson_id).limit(1).execute().data
        )
        previous = existing[0] if existing else {}
        old_best = previous.get("best_score")
        best = score if old_best is None else max(float(old_best), score)
        already_passed = previous.get("status") == "passed"
        row = {
            "user_id": user_id,
            "lesson_id": lesson_id,
            "status": "passed" if (passed or already_passed) else "in_progress",
            "best_score": round(best, 3),
            "tests_taken": int(previous.get("tests_taken") or 0) + 1,
        }
        saved = self._table("lesson_progress").upsert(row, on_conflict="user_id,lesson_id").execute().data
        return saved[0] if saved else row

    # ---------- attempts ----------

    # Inserts one written answer into the attempts table and returns the saved row.
    def record_attempt(
        self,
        user_id: str,
        lesson_id: str,
        target: str,
        ocr_text: str | None,
        ocr_disputed: bool,
        letters_correct: bool | None,
        harakat_input: list[dict[str, Any]] | None,
        harakat_correct: bool | None,
        is_correct: bool | None,
        counted: bool,
        errors: list[dict[str, Any]],
        reveal_count: int,
        image_path: str | None,
        strokes_path: str | None,
    ) -> dict[str, Any]:
        row = {
            "user_id": user_id,
            "lesson_id": lesson_id,
            "target": target,
            "ocr_text": ocr_text,
            "ocr_disputed": ocr_disputed,
            "letters_correct": letters_correct,
            "harakat_input": harakat_input,
            "harakat_correct": harakat_correct,
            "is_correct": is_correct,
            "counted": counted,
            "errors": errors,
            "reveal_count": reveal_count,
            "image_path": image_path,
            "strokes_path": strokes_path,
        }
        saved = self._table("attempts").insert(row).execute().data
        return saved[0] if saved else row

    # ---------- storage ----------

    # Uploads the answer PNG and stroke JSON under "{user_id}/" and returns both paths.
    def upload_attempt_files(self, user_id: str, png_bytes: bytes, strokes_json: dict | str) -> tuple[str, str]:
        file_id = uuid.uuid4().hex
        image_path = f"{user_id}/{file_id}.png"
        strokes_path = f"{user_id}/{file_id}.json"
        strokes_text = strokes_json if isinstance(strokes_json, str) else json.dumps(strokes_json)
        bucket = self._bucket()
        bucket.upload(image_path, png_bytes, {"content-type": "image/png"})
        bucket.upload(strokes_path, strokes_text.encode("utf-8"), {"content-type": "application/json"})
        return image_path, strokes_path

    # Downloads a stored answer PNG so the OCR can read it.
    def download_attempt_image(self, image_path: str) -> bytes:
        return self._bucket().download(image_path)


class AdminDatabase(Database):
    # Admin-only client built from SUPABASE_SECRET_KEY (.env.admin); only build_lesson_cards.py creates it.
    def __init__(self) -> None:
        secret = os.environ.get("SUPABASE_SECRET_KEY")
        if not secret:
            raise RuntimeError("Set SUPABASE_SECRET_KEY in .env.admin")
        super().__init__(api_key=secret)

    # Inserts or updates lesson cards (lesson_id, card_text, sample_questions, embedding) by lesson_id.
    def upsert_lesson_cards(self, rows: list[dict[str, Any]]) -> None:
        self.client.table("lesson_cards").upsert(rows, on_conflict="lesson_id").execute()
