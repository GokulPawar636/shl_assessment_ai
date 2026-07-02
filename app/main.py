from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.conversation_manager import ConversationManager

app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Lazy Load Conversation Manager
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_manager() -> ConversationManager:
    """
    Creates the ConversationManager only once.

    The first request initializes it.
    Every later request reuses the same instance.
    """
    return ConversationManager()


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "SHL Assessment Recommendation Agent Running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ---------------------------------------------------------------------
# Chat Endpoint
# ---------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    manager = get_manager()

    try:
        result = manager.handle_history(
            [m.model_dump() for m in req.messages]
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        reply=result.reply,
        recommendations=_to_public_recommendations(
            result.recommendations or []
        ),
        end_of_conversation=result.end_of_conversation,
    )


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def _to_public_recommendations(
    rows: list[dict],
) -> list[RecommendationItem]:

    recommendations = []

    for row in rows[:10]:

        recommendations.append(
            RecommendationItem(
                name=row["name"],
                url=row["url"],
                test_type=row["test_type"],
            )
        )

    return recommendations