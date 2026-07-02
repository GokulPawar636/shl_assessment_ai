from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.conversation_manager import ConversationManager

app = FastAPI(title="SHL Assessment Recommendation Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConversationManager()


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationItem]
    end_of_conversation: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = manager.handle_history([m.model_dump() for m in req.messages])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatResponse(
        reply=result.reply,
        recommendations=_to_public_recommendations(result.recommendations or []),
        end_of_conversation=result.end_of_conversation,
    )


def _to_public_recommendations(rows: list[dict]) -> list[RecommendationItem]:
    public_rows = []
    for row in rows[:10]:
        public_rows.append(
            RecommendationItem(
                name=row["name"],
                url=row["url"],
                test_type=row["test_type"],
            )
        )
    return public_rows
