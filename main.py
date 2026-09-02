
import json
import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq, GroqError
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_resume_screener")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are an expert technical recruiter that analyzes a candidate's resume "
    "against a job description. Compare the resume skills against the required "
    "and preferred skills in the job description. Then respond with ONLY a "
    "single valid JSON object and nothing else. No markdown, no code fences, no "
    "explanatory text.\n"
    "The JSON must have exactly this shape:\n"
    '{"match_score": 0, "matched_skills": ["Python"], "missing_skills": ["AWS"]}\n'
    "Rules:\n"
    "- match_score: integer 0-100 reflecting overall fit.\n"
    "- matched_skills: skills in the resume that satisfy the job description.\n"
    "- missing_skills: skills required/preferred by the job description that are "
    "absent from the resume.\n"
    "- Only list a skill if there is clear evidence in the resume text."
)


class MatchRequest(BaseModel):
    job_description: str
    resume: str


class MatchResponse(BaseModel):
    match_score: int = Field(..., ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


app = FastAPI(title="AI Resume Screener", version="1.0.0")

_available: Optional[Groq] = None
_missing_api_key_warned = False


def get_client() -> Optional[Groq]:
    """Return the shared Groq client, or None if no API key is configured."""
    global _available, _missing_api_key_warned
    if GROQ_API_KEY:
        if _available is None:
            _available = Groq(api_key=GROQ_API_KEY)
        return _available
    if not _missing_api_key_warned:
        _missing_api_key_warned = True
        logger.warning(
            "GROQ_API_KEY is not set. Set it in a .env file or environment."
        )
    return None


def strip_reasoning_tags(text: str) -> str:
    """Remove common reasoning blocks that models sometimes prepend."""
    text = re.sub(r"<\s*thinking\s*>.*?<\s*/\s*thinking\s*>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>", "", text, flags=re.DOTALL)
    return text


def strip_markdown_fences(text: str) -> str:
    """Remove Markdown code fences (```json ... ``` or ``` ... ```)."""
    return re.sub(r"```[a-zA-Z]*\s*|\s*```", "", text).strip()


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Safely extract a JSON object from a model response.

    Handles reasoning tags, Markdown fences, and surrounding prose by isolating
    the first {...} block and parsing it. Returns None when parsing fails.
    """
    if not text:
        return None

    cleaned = strip_markdown_fences(strip_reasoning_tags(text))

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    candidate = cleaned[start : end + 1]

    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        return None

    return None


def build_response(data: dict[str, Any]) -> MatchResponse:
    """Coerce a parsed dict into a validated MatchResponse."""
    score = data.get("match_score", 0)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    def _to_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item.strip())
        return out

    return MatchResponse(
        match_score=score,
        matched_skills=_to_str_list(data.get("matched_skills")),
        missing_skills=_to_str_list(data.get("missing_skills")),
    )


def build_prompt(request: MatchRequest) -> str:
    return (
        "Job Description:\n"
        "---\n"
        f"{request.job_description}\n"
        "---\n\n"
        "Candidate Resume:\n"
        "---\n"
        f"{request.resume}\n"
        "---\n\n"
        "Return the JSON object only."
    )


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "AI Resume Screener",
        "endpoints": {"POST": "/test-match"},
        "model": GROQ_MODEL,
        "api_key_set": bool(GROQ_API_KEY),
    }


@app.post("/test-match", response_model=MatchResponse)
def test_match(request: MatchRequest) -> MatchResponse:
    """Score a resume against a job description via a Groq model."""
    client = get_client()
    if client is None:
        raise HTTPException(
            status_code=502,
            detail="GROQ_API_KEY is not configured on the server.",
        )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(request)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except GroqError as exc:
        logger.error("Groq API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Groq API error: {exc}")

    raw_text = completion.choices[0].message.content or ""

    data = extract_json(raw_text)
    if data is None:
        logger.error("Could not parse model output: %r", raw_text[:1000])
        raise HTTPException(
            status_code=500,
            detail="Model returned an unparseable JSON response.",
        )

    return build_response(data)
