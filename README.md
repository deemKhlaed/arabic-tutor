# Arabic spelling tutor (إملاء)

A Streamlit chat app that teaches Arabic spelling to Arabic-speaking children and non-native speakers.
A LangGraph agent teaches each lesson, then tests the learner. The learner writes each answer on a canvas.
A Cloudflare vision model reads the letters, the learner taps the harakat, and progress is saved in Supabase.

## Files

| File | Job |
|---|---|
| `Database.py` | Every Supabase call: auth, profiles, lessons, rules, progress, attempts, storage |
| `model.py` | Cloudflare Workers AI models: chat and handwriting OCR, both Qwen 3.8 27B |
| `tools.py` | Pure helpers: canvas to PNG, text normalization, letter/harakat comparison, test items, prompts |
| `Nodes.py` | Every LangGraph node and router (`TutorNodes`, with `db` and `models` injected) |
| `Main.py` | Builds the graph and runs the Streamlit UI |
| `test_graph.py` | Tests every graph step and, with `--live`, runs the routing evaluation; prints reports, plots charts |
| `build_lesson_cards.py` | Admin script: checks the sample questions, embeds one card per lesson, upserts `lesson_cards` |
| `data/lesson_cards.json` | 10 sample questions per lesson (dialect, standard Arabic, English) |
| `data/routing_eval.jsonl` | 129 routing cases for the live evaluation |

## Setup

1. **Python 3.11 or newer.** Tested on 3.14.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment.** Copy `.env.example` to `.env` and fill it in.

   | Variable | What it is |
   |---|---|
   | `CF_ACCOUNT_ID`, `CF_AI_API_TOKEN` | Cloudflare account and a Workers AI token (`CF_API_TOKEN` also works) |
   | `CF_MODEL` | Chat model (default `@cf/qwen/qwen3.8-27b`) |
   | `CF_CHAT_REASONING` | Qwen reasoning: `off` (default, replies in 2–4 s) or `low` / `medium` (slower) |
   | `CF_VISION_MODEL` | OCR model (default `@cf/qwen/qwen3.8-27b`; `@cf/meta/llama-3.2-11b-vision-instruct` and `@cf/moondream/...` also work) |
   | `SUPABASE_URL` | Project URL |
   | `SUPABASE_ANON_KEY` | The anon / publishable key (`SUPABASE_PUBLISHABLE_KEY` is also read). **Never** the service_role key. |
   | `SUPABASE_DB_URL` | Postgres connection string for the LangGraph checkpointer. Use the **session pooler (port 5432)** or the direct connection. If unset, sessions are kept in memory and a warning is shown. |
   | `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` | Optional tracing. EU accounts must set `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`. Runs are tagged with `user_id`, `lesson_id` and `mode`. |

3. **Database.** The Supabase project already has the learner tables (`profiles`, `lesson_progress`, `attempts`,
   with row-level security), the private storage bucket `attempts`, and the pgvector table `lesson_cards` with the
   `match_lessons` function. The curriculum tables (`levels`, `lessons`, `rules`) must be readable with the anon key.

4. **Only if you switch the OCR to Llama 3.2 Vision: accept its license (once per Cloudflare account).** It needs a one-time
   `{"prompt": "agree"}` request. Sending it means **you** accept the
   [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)
   and the Acceptable Use Policy. It also means you confirm you are not an individual or company based in the EU.

   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(); from model import Models; print(Models().accept_vision_license())"
   ```

4b. **Lesson cards (admin, once and after editing `data/lesson_cards.json`).** Put `SUPABASE_URL`,
   `SUPABASE_SECRET_KEY` and the Cloudflare variables in `.env.admin` (see `.env.admin.example`). Then:

   ```bash
   python build_lesson_cards.py --check   # flags questions that fit another lesson as well as their own
   python build_lesson_cards.py           # check, embed with bge-m3, upsert into lesson_cards
   ```

## Run the app

```bash
streamlit run Main.py
```

The flow is: sign up, then onboarding (learner type, explanation language, start level), then learn mode, then
"I'm ready". In test mode the item shows on a timed card, and the learner writes it on the canvas and presses
Submit. The app then shows "I read: …" for confirmation. On levels 1–3, the learner next taps the harakat.
Each item gets feedback, and the test ends with a score. A pass moves the learner to the next lesson.

Each learner gets one graph thread per day (`{user_id}:{YYYY-MM-DD}`), stored in Postgres. After a page reload,
sign in again and the same day's session resumes where it stopped.

## Test

```bash
python test_graph.py            # fakes only: every node, router and helper, plus an end-to-end walk
python test_graph.py --live     # also: Supabase read, real OCR, real chat, Postgres checkpointer
python test_graph.py --no-plot  # skip the chart window
```

The script prints a `step | PASS/FAIL | ms | detail` table, then shows a bar chart of the time per step.
It exits with code 1 if any step fails. Set `TEST_VERBOSE=1` for tracebacks.

## Design notes

- **Interface language.** Children get Arabic. Non-native learners get their explanation language: Arabic and
  English are built in, and any other language is translated once by the chat model. Right-to-left languages
  get a right-to-left layout.
- **Chat during the test.** The learner can ask a question at any test step. The examiner answers without
  giving away the item, then returns to the same step. If the learner asks to stop or go back to the lesson,
  the test ends without a score and the teacher takes over.
- **Canvas toolbar.** The ✎ button above the canvas is the library's *edit/select* toggle, not the pen.
  While it is on, the pen does not draw. Click it again, or press Clear.

- **Lesson content comes from Supabase.** Test items are built only from `target_items`, rule examples and
  `dictation_word` exercises. The teacher may explain and transliterate.
- **Changing lesson by chat (RAG).** Every chat message is embedded with bge-m3 and matched against one card
  per lesson in Supabase (`match_lessons`, top 3). If the current lesson leads clearly (similarity ≥ `MIN_SIM`,
  ahead by ≥ `MARGIN`, both in `tools.py`), the message is about this lesson and no LLM call is made. Otherwise one
  router call (8 s timeout) picks `stay`, `switch`, `ready`, `chat` or `stop`; the next lesson is always a candidate
  so "something harder" works. Anything unparseable becomes `stay`. A switch is confirmed with buttons in the chat
  (two lesson buttons + "Neither" when unsure); in a test the learner chooses "Finish the test first" or
  "Stop and switch" (stopping saves no score). Unpassed prerequisites are mentioned but never block a switch.
  Every decision is logged to LangSmith as a `route_decision` run with its candidates and scores.
- **The OCR checks letters only.** Its harakat are stripped. On levels 1–3 the harakat come from the learner's taps,
  using `compare_harakat` and `render_harakat_input` in `Main.py`. Either one can later be swapped for a trained model.
- **Disputed reads.** If the learner says the OCR misread their answer, the app logs that attempt with
  `counted=false` and gives a fresh canvas. After a second dispute the item is skipped and doesn't affect the score.
- **Images stay out of checkpoints.** The UI uploads the PNG and stroke JSON to storage, then resumes the graph
  with their paths only.
