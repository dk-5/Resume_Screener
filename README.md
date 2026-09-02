# AI Resume Screener

A standalone FastAPI service that scores a candidate's resume against a job
description using a Groq-hosted, open-weight language model.

The model returns a match score plus the skills the candidate has (matched) and
the skills they're missing. Model output is parsed defensively so it still works
even if the model wraps its JSON in Markdown code fences (```json ... ```) or
emits reasoning tags (<thinking>...</thinking>).

## Features

- `POST /test-match` — takes a job description and a plain-text resume, returns:
  - `match_score` (int 0–100)
  - `matched_skills` (list)
  - `missing_skills` (list)
- Robust JSON extraction (handles Markdown fences, reasoning tags, prose)
- Clean error handling with meaningful HTTP status codes
- Model configurable via environment variable

## Requirements

- Python 3.10+
- A Groq API key (free tier; no credit card required) from https://console.groq.com

## Setup

```powershell
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set your key:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

`GROQ_MODEL` defaults to `openai/gpt-oss-120b` (free-tier, open-weight, currently
supported). Other options include `openai/gpt-oss-20b` (faster/cheaper) and
`qwen/qwen3.6-27b` (strong reasoning).

## Run

```powershell
uvicorn main:app --reload
```

- API: http://127.0.0.1:8000
- Interactive docs (Swagger UI): http://127.0.0.1:8000/docs

## Usage

### Swagger UI (easiest)

Open http://127.0.0.1:8000/docs, expand `POST /test-match`, click **Try it out**,
paste text into the `job_description` and `resume` fields, and hit **Execute**.

### curl with a JSON file

Multiline text must be valid JSON (newlines escaped as `\n`). The simplest way is
to put the payload in a file and reference it:

```powershell
curl -X POST http://127.0.0.1:8000/test-match `
  -H "Content-Type: application/json" `
  -d @payload.json
```

Example `payload.json`:

```json
{
  "job_description": "Backend Software Engineer. Need Node.js, TypeScript, PostgreSQL...",
  "resume": "Backend Software Engineer. Experienced with Node.js, TypeScript, MongoDB..."
}
```



### Health

```
GET /   -> service info, model name, and whether an API key is set
```

## Error handling

| Situation                          | HTTP status |
| ---------------------------------- | ----------- |
| Missing/invalid `GROQ_API_KEY`     | 502         |
| Groq API failure                   | 502         |
| Model returned unparseable JSON    | 500         |
| Malformed request body             | 422         |

Note: if you send raw newlines inside a JSON string (e.g. pasting a resume into
a `curl -d "..."` argument), the request body is invalid JSON and you'll get a
422. Use the Swagger UI or a JSON file (see above) instead.
