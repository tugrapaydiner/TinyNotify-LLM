"""
FastAPI server for real-time notification scoring.
Provides REST API endpoints for the TinyNotify-LLM system.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import uvicorn
import logging
import pickle

from src.config import Config, DEFAULT_CONFIG
from src.features.schema import UserFeatures, CandidateNotification, Channel, NotificationType
from src.models.base_model import BaseClickModel, BaseComplaintModel
from src.models.fatigue_model import FatigueModel
from src.llm.client import MockLLMClient, TinyLLMClient
from src.policy.scoring import NotificationScorer
from src.policy.decision import NotificationDecisionMaker

logger = logging.getLogger(__name__)


# Pydantic models for API
class UserFeaturesInput(BaseModel):
    """Input model for user features."""
    user_id: int
    notif_count_24h: int = 0
    notif_count_7d: int = 0
    notif_count_30d: int = 0
    click_count_7d: int = 0
    click_count_30d: int = 0
    open_rate_7d: float = 0.0
    open_rate_30d: float = 0.0
    hours_since_last_notification: Optional[float] = None
    hours_since_last_click: Optional[float] = None
    most_active_hour_bucket: Optional[int] = None
    complaint_count_30d: int = 0


class NotificationInput(BaseModel):
    """Input model for a candidate notification."""
    template_id: str
    channel: str = Field(..., description="Channel: email, push, sms, in_app")
    notification_type: str = Field(..., description="Type: promo, product_tip, important, system_alert")
    text: str = Field(..., max_length=500)
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)


class DecisionRequest(BaseModel):
    """Request to make notification decision."""
    user_features: UserFeaturesInput
    candidate: NotificationInput
    use_llm: bool = False


class BatchDecisionRequest(BaseModel):
    """Request to make decisions for multiple candidates."""
    user_features: UserFeaturesInput
    candidates: List[NotificationInput]
    use_llm_for_top_k: bool = False
    top_k: Optional[int] = None


class DecisionResponse(BaseModel):
    """Response with decision."""
    should_send: bool
    final_score: float
    p_click: float
    p_complaint: float
    base_score: float
    llm_rating: Optional[int] = None
    fatigue_penalty: float
    reason: str
    template_id: str


class BatchDecisionResponse(BaseModel):
    """Response with batch decisions."""
    decisions: List[DecisionResponse]
    stats: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    llm_available: bool
    memory_mb: float


class ModelInfo(BaseModel):
    """Model information."""
    model_type: str
    trained: bool
    feature_count: Optional[int] = None


# Initialize FastAPI
app = FastAPI(
    title="TinyNotify-LLM API",
    description="Real-time notification scoring and decision API",
    version="0.1.0"
)

# Global state
config: Config = DEFAULT_CONFIG
click_model: Optional[BaseClickModel] = None
complaint_model: Optional[BaseComplaintModel] = None
fatigue_model: Optional[FatigueModel] = None
llm_client: Optional[TinyLLMClient] = None
decision_maker: Optional[NotificationDecisionMaker] = None


def load_models_from_disk():
    """Load trained models from disk."""
    global click_model, complaint_model

    click_model_path = config.paths.models_dir / "click_model.pkl"
    complaint_model_path = config.paths.models_dir / "complaint_model.pkl"

    if click_model_path.exists():
        with open(click_model_path, 'rb') as f:
            click_model = pickle.load(f)
        logger.info("✓ Click model loaded")

    if complaint_model_path.exists():
        with open(complaint_model_path, 'rb') as f:
            complaint_model = pickle.load(f)
        logger.info("✓ Complaint model loaded")


def initialize_components():
    """Initialize all components."""
    global fatigue_model, llm_client, decision_maker

    # Initialize fatigue model
    fatigue_model = FatigueModel(config)

    # Initialize LLM client (use mock by default)
    llm_client = MockLLMClient(config)
    llm_client.load_model()

    # Initialize decision maker if models are loaded
    if click_model is not None and complaint_model is not None:
        decision_maker = NotificationDecisionMaker(
            click_model=click_model,
            complaint_model=complaint_model,
            fatigue_model=fatigue_model,
            llm_client=llm_client,
            config=config
        )
        logger.info("✓ Decision maker initialized")


@app.on_event("startup")
async def startup_event():
    """Load models and initialize components on startup."""
    logger.info("Starting TinyNotify-LLM API...")

    # Try to load models from disk
    try:
        load_models_from_disk()
    except Exception as e:
        logger.warning(f"Could not load models from disk: {e}")
        logger.warning("API will run with untrained models for testing only")

    # Initialize components
    initialize_components()

    # Memory check
    import psutil
    import os
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage: {mem_mb:.1f} MB")

    logger.info("✓ API ready")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "TinyNotify-LLM",
        "version": "0.1.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/models/info",
            "/decide",
            "/decide_batch",
            "/docs"
        ]
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    import psutil
    import os

    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    return HealthResponse(
        status="healthy" if decision_maker is not None else "degraded",
        models_loaded=click_model is not None and complaint_model is not None,
        llm_available=llm_client is not None,
        memory_mb=mem_mb
    )


@app.get("/models/info")
async def models_info():
    """Get information about loaded models."""
    return {
        "click_model": ModelInfo(
            model_type="BaseClickModel",
            trained=click_model is not None and click_model.model is not None,
            feature_count=len(click_model.feature_names) if click_model and hasattr(click_model, 'feature_names') else None
        ).dict(),
        "complaint_model": ModelInfo(
            model_type="BaseComplaintModel",
            trained=complaint_model is not None and complaint_model.model is not None,
            feature_count=len(complaint_model.feature_names) if complaint_model and hasattr(complaint_model, 'feature_names') else None
        ).dict(),
        "fatigue_model": ModelInfo(
            model_type="FatigueModel",
            trained=True  # Rule-based, always available
        ).dict(),
        "llm_client": {
            "type": llm_client.__class__.__name__ if llm_client else None,
            "available": llm_client is not None
        }
    }


def convert_user_features_input(user_input: UserFeaturesInput) -> UserFeatures:
    """Convert API input to UserFeatures object."""
    return UserFeatures(
        user_id=user_input.user_id,
        timestamp=datetime.now(),
        notif_count_24h=user_input.notif_count_24h,
        notif_count_7d=user_input.notif_count_7d,
        notif_count_30d=user_input.notif_count_30d,
        click_count_7d=user_input.click_count_7d,
        click_count_30d=user_input.click_count_30d,
        open_rate_7d=user_input.open_rate_7d,
        open_rate_30d=user_input.open_rate_30d,
        hours_since_last_notification=user_input.hours_since_last_notification,
        hours_since_last_click=user_input.hours_since_last_click,
        most_active_hour_bucket=user_input.most_active_hour_bucket,
        complaint_count_30d=user_input.complaint_count_30d
    )


def convert_notification_input(
    notif_input: NotificationInput,
    user_features: UserFeatures
) -> CandidateNotification:
    """Convert API input to CandidateNotification object."""
    return CandidateNotification(
        user_id=user_features.user_id,
        template_id=notif_input.template_id,
        channel=Channel(notif_input.channel),
        notification_type=NotificationType(notif_input.notification_type),
        text=notif_input.text,
        timestamp=datetime.now(),
        hour_of_day=notif_input.hour_of_day,
        day_of_week=notif_input.day_of_week,
        user_features=user_features
    )


def format_decision_response(decision) -> DecisionResponse:
    """Format Decision object as API response."""
    return DecisionResponse(
        should_send=decision.should_send,
        final_score=decision.scored.final_score,
        p_click=decision.scored.p_click,
        p_complaint=decision.scored.p_complaint,
        base_score=decision.scored.base_score,
        llm_rating=decision.scored.llm_rating,
        fatigue_penalty=decision.scored.fatigue_penalty,
        reason=decision.reason,
        template_id=decision.candidate.template_id
    )


@app.post("/decide", response_model=DecisionResponse)
async def decide_single(request: DecisionRequest):
    """
    Make send/no-send decision for a single notification candidate.

    Args:
        request: DecisionRequest with user features and candidate

    Returns:
        DecisionResponse with decision and scores
    """
    if decision_maker is None:
        raise HTTPException(
            status_code=503,
            detail="Decision maker not initialized. Models may not be loaded."
        )

    try:
        # Convert input to internal format
        user_features = convert_user_features_input(request.user_features)
        candidate = convert_notification_input(request.candidate, user_features)

        # Make decision
        decision = decision_maker.decide_single(candidate, use_llm=request.use_llm)

        # Format response
        return format_decision_response(decision)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Error making decision: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/decide_batch", response_model=BatchDecisionResponse)
async def decide_batch(request: BatchDecisionRequest):
    """
    Make decisions for multiple notification candidates.

    Uses efficient batch processing with optional LLM for top-K.

    Args:
        request: BatchDecisionRequest with user features and candidates

    Returns:
        BatchDecisionResponse with decisions and statistics
    """
    if decision_maker is None:
        raise HTTPException(
            status_code=503,
            detail="Decision maker not initialized. Models may not be loaded."
        )

    try:
        # Convert input to internal format
        user_features = convert_user_features_input(request.user_features)

        candidates = []
        for notif_input in request.candidates:
            candidate = convert_notification_input(notif_input, user_features)
            candidates.append(candidate)

        # Make batch decisions
        decisions = decision_maker.decide_batch(
            candidates,
            use_llm_for_top_k=request.use_llm_for_top_k,
            top_k=request.top_k
        )

        # Format responses
        decision_responses = [format_decision_response(d) for d in decisions]

        # Compute stats
        stats = decision_maker.get_decision_stats(decisions)

        return BatchDecisionResponse(
            decisions=decision_responses,
            stats=stats
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Error making batch decisions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/train")
async def train_models(data_path: str):
    """
    Trigger model training (for admin use).

    Args:
        data_path: Path to training data

    Returns:
        Training status
    """
    # This is a simplified endpoint - in production would be more sophisticated
    raise HTTPException(
        status_code=501,
        detail="Training endpoint not yet implemented. Use CLI or dashboard for training."
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == '__main__':
    run_server()
