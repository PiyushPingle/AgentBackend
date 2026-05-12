
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from groq import AsyncGroq
from dotenv import load_dotenv
from typing import Optional, List, Dict
from PIL import Image
import pytesseract
import io
import os

load_dotenv()

app = FastAPI(title="GenAI Chatbot API (Groq)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -----------------------------
# GROQ CLIENT
# -----------------------------
client = AsyncGroq(api_key="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")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and clear."

# -----------------------------
# SCHEMAS
# -----------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[Dict]] = []
    temperature: Optional[float] = 0.7
    model: Optional[str] = DEFAULT_MODEL

class ChatResponse(BaseModel):
    reply: str
    tokens_used: int

# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "provider": "Groq",
        "default_model": DEFAULT_MODEL
    }

# -----------------------------
# NORMAL CHAT
# -----------------------------
@app.post("/ask", response_model=ChatResponse)
async def ask(req: ChatRequest):

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if req.history:
        messages.extend(req.history)

    messages.append({"role": "user", "content": req.message})

    resp = await client.chat.completions.create(
        model=req.model,
        messages=messages,
        temperature=req.temperature,
    )

    return ChatResponse(
        reply=resp.choices[0].message.content,
        tokens_used=resp.usage.total_tokens
    )

# -----------------------------
# STREAMING
# -----------------------------
@app.post("/stream")
async def stream_ask(req: ChatRequest):

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": req.message})

    async def token_generator():
        stream = await client.chat.completions.create(
            model=req.model,
            messages=messages,
            temperature=req.temperature,
            stream=True,
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return StreamingResponse(token_generator(), media_type="text/plain")

# -----------------------------
# IMAGE OCR + LLM
# -----------------------------
@app.post("/image")
async def image_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # OCR
        text = pytesseract.image_to_string(image)

        # DEBUG PRINT
        print("\n--- OCR TEXT ---\n", text)

        # अगर text empty hai to fallback
        if not text.strip():
            return {
                "extracted_text": "",
                "answer": "⚠️ No readable text found in image. Try clearer image."
            }

        # LLM call
        messages = [
            {
                "role": "system",
                "content": "You are a marketing expert. Analyze the given content and give insights."
            },
            {
                "role": "user",
                "content": text
            }
        ]

        resp = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.7,
        )

        answer = resp.choices[0].message.content

        return {
            "extracted_text": text,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)