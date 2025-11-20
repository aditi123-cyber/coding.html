import os
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
import uuid
import re
import asyncio

# Load environment variables
load_dotenv()

# ---------------- CONFIG ----------------
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY not set. Add your Gemini API key to .env")

# Build full Gemini URL
LLM_API_URL = f"https://generativelanguage.googleapis.com/v1/models/{LLM_MODEL}:generateContent?key={LLM_API_KEY}"

# ---------------- FASTAPI SETUP ----------------
app = FastAPI(title="FixItNow API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>index.html not found in /static</h1>")

# ---------------- PROMPT BUILDER ----------------
def build_repair_prompt(user_text: str, image_hint: Optional[str] = None) -> str:
    system = (
        "You are RepairGPT — an expert technician. "
        "Return ONLY valid JSON with these fields:\n"
        "diagnosis, steps, tools, risk_level, estimated_cost, time_minutes,\n"
        "when_to_call_pro, quick_check, safety_notes.\n"
        "No explanations outside JSON. Keep responses practical and concise."
    )
    user = f"Problem: {user_text.strip()}"
    if image_hint:
        user += f" | Image hint: {image_hint}"
    return system + "\n" + user

# ---------------- GEMINI API CALL ----------------
async def call_llm(prompt: str) -> dict:
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(LLM_API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Gemini API error: {e.response.text}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Request failed: {str(e)}") from e

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidates returned: {data}")

    content = candidates[0]["content"]["parts"][0]["text"]

    # Parse JSON robustly
    try:
        return json.loads(content)
    except:
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Gemini returned invalid JSON:\n" + content)

# ---------------- ENDPOINTS ----------------
@app.post("/fix")
async def fix(problem: str = Form(...)):
    prompt = build_repair_prompt(problem)
    try:
        result = await call_llm(prompt)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/fix-image")
async def fix_image(problem: str = Form(...), file: UploadFile = File(None)):
    image_hint = None
    saved_path = None

    if file:
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        saved_path = f"uploads/{uuid.uuid4().hex}{ext}"
        content = await file.read()
        with open(saved_path, "wb") as f:
            f.write(content)
        image_hint = f"File: {saved_path}, size_kb={len(content)//1024}"

    prompt = build_repair_prompt(problem, image_hint)

    try:
        result = await call_llm(prompt)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    if saved_path:
        result["_image_saved"] = saved_path

    return JSONResponse(result)

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
