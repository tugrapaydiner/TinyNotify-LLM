"""
Tests for FastAPI server.
Tests all API endpoints with various scenarios.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.server import app
from src.models.base_model import BaseClickModel, BaseComplaintModel
from src.models.fatigue_model import FatigueModel
from src.llm.client import MockLLMClient
from src.policy.decision import NotificationDecisionMaker


# Create test client
client = TestClient(app)


@pytest.fixture
def mock_decision_maker():
    """Create a mock decision maker for testing."""
    from tests.test_policy import create_test_decision_maker
    return create_test_decision_maker()


def test_root_endpoint():
    """Test root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "TinyNotify-LLM"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"
    assert "endpoints" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "llm_available" in data
    assert "memory_mb" in data
    assert data["memory_mb"] > 0


def test_models_info_endpoint():
    """Test models info endpoint."""
    response = client.get("/models/info")
    assert response.status_code == 200

    data = response.json()
    assert "click_model" in data
    assert "complaint_model" in data
    assert "fatigue_model" in data
    assert "llm_client" in data

    # Check structure
    assert "model_type" in data["click_model"]
    assert "trained" in data["click_model"]


def test_decide_endpoint_without_models():
    """Test decide endpoint when models are not loaded."""
    request_data = {
        "user_features": {
            "user_id": 123,
            "notif_count_24h": 2,
            "notif_count_7d": 10,
            "notif_count_30d": 30,
            "open_rate_7d": 0.15,
            "open_rate_30d": 0.12
        },
        "candidate": {
            "template_id": "promo_001",
            "channel": "push",
            "notification_type": "promo",
            "text": "Special offer just for you!",
            "hour_of_day": 14,
            "day_of_week": 2
        },
        "use_llm": False
    }

    response = client.post("/decide", json=request_data)

    # May return 503 if models not loaded, which is expected
    # Or 200 if mock models are initialized
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "should_send" in data
        assert "final_score" in data
        assert "p_click" in data
        assert "p_complaint" in data
        assert "reason" in data
        assert "template_id" in data
        assert data["template_id"] == "promo_001"


def test_decide_endpoint_with_trained_models(tmp_path):
    """Test decide endpoint with trained models."""
    # Import test utilities
    from tests.test_base_model import create_synthetic_dataset
    from src.api import server

    # Create and train models
    df = create_synthetic_dataset(n_samples=500)

    click_model = BaseClickModel()
    click_model.train(df, target_col='clicked', verbose=False)

    complaint_model = BaseComplaintModel()
    complaint_model.train(df, target_col='unsubscribed', verbose=False)

    fatigue_model = FatigueModel()
    llm_client = MockLLMClient()
    llm_client.load_model()

    # Inject trained models into server
    server.click_model = click_model
    server.complaint_model = complaint_model
    server.fatigue_model = fatigue_model
    server.llm_client = llm_client

    server.decision_maker = NotificationDecisionMaker(
        click_model=click_model,
        complaint_model=complaint_model,
        fatigue_model=fatigue_model,
        llm_client=llm_client
    )

    # Make request
    request_data = {
        "user_features": {
            "user_id": 123,
            "notif_count_24h": 2,
            "notif_count_7d": 10,
            "notif_count_30d": 30,
            "click_count_7d": 3,
            "click_count_30d": 8,
            "open_rate_7d": 0.15,
            "open_rate_30d": 0.12
        },
        "candidate": {
            "template_id": "promo_001",
            "channel": "push",
            "notification_type": "promo",
            "text": "Special offer just for you!",
            "hour_of_day": 14,
            "day_of_week": 2
        },
        "use_llm": False
    }

    response = client.post("/decide", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["should_send"], bool)
    assert isinstance(data["final_score"], float)
    assert isinstance(data["p_click"], float)
    assert isinstance(data["p_complaint"], float)
    assert 0 <= data["p_click"] <= 1
    assert 0 <= data["p_complaint"] <= 1
    assert data["template_id"] == "promo_001"
    assert len(data["reason"]) > 0


def test_decide_batch_endpoint_with_trained_models(tmp_path):
    """Test batch decide endpoint with trained models."""
    from tests.test_base_model import create_synthetic_dataset
    from src.api import server

    # Create and train models
    df = create_synthetic_dataset(n_samples=500)

    click_model = BaseClickModel()
    click_model.train(df, target_col='clicked', verbose=False)

    complaint_model = BaseComplaintModel()
    complaint_model.train(df, target_col='unsubscribed', verbose=False)

    fatigue_model = FatigueModel()
    llm_client = MockLLMClient()
    llm_client.load_model()

    # Inject models
    server.click_model = click_model
    server.complaint_model = complaint_model
    server.fatigue_model = fatigue_model
    server.llm_client = llm_client

    server.decision_maker = NotificationDecisionMaker(
        click_model=click_model,
        complaint_model=complaint_model,
        fatigue_model=fatigue_model,
        llm_client=llm_client
    )

    # Make batch request
    request_data = {
        "user_features": {
            "user_id": 123,
            "notif_count_24h": 2,
            "notif_count_7d": 10,
            "notif_count_30d": 30,
            "click_count_7d": 3,
            "click_count_30d": 8,
            "open_rate_7d": 0.15,
            "open_rate_30d": 0.12
        },
        "candidates": [
            {
                "template_id": "promo_001",
                "channel": "push",
                "notification_type": "promo",
                "text": "Special offer!",
                "hour_of_day": 14,
                "day_of_week": 2
            },
            {
                "template_id": "product_tip_002",
                "channel": "email",
                "notification_type": "product_tip",
                "text": "Check out this feature",
                "hour_of_day": 10,
                "day_of_week": 3
            },
            {
                "template_id": "system_alert_003",
                "channel": "in_app",
                "notification_type": "system_alert",
                "text": "Important update",
                "hour_of_day": 9,
                "day_of_week": 1
            }
        ],
        "use_llm_for_top_k": True,
        "top_k": 2
    }

    response = client.post("/decide_batch", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "decisions" in data
    assert "stats" in data

    # Check decisions
    assert len(data["decisions"]) == 3

    for decision in data["decisions"]:
        assert isinstance(decision["should_send"], bool)
        assert isinstance(decision["final_score"], float)
        assert "template_id" in decision
        assert "reason" in decision

    # Check stats
    stats = data["stats"]
    assert stats["total"] == 3
    assert stats["sent"] + stats["rejected"] == 3
    assert 0 <= stats["send_rate"] <= 1


def test_decide_endpoint_invalid_channel():
    """Test decide endpoint with invalid channel."""
    request_data = {
        "user_features": {
            "user_id": 123,
            "notif_count_24h": 2,
            "notif_count_7d": 10,
            "notif_count_30d": 30
        },
        "candidate": {
            "template_id": "promo_001",
            "channel": "invalid_channel",  # Invalid
            "notification_type": "promo",
            "text": "Test",
            "hour_of_day": 14,
            "day_of_week": 2
        }
    }

    response = client.post("/decide", json=request_data)

    # Should return 400 or 422 for invalid input
    assert response.status_code in [400, 422, 503]


def test_decide_endpoint_invalid_hour():
    """Test decide endpoint with invalid hour."""
    request_data = {
        "user_features": {
            "user_id": 123,
            "notif_count_24h": 2,
            "notif_count_7d": 10,
            "notif_count_30d": 30
        },
        "candidate": {
            "template_id": "promo_001",
            "channel": "push",
            "notification_type": "promo",
            "text": "Test",
            "hour_of_day": 25,  # Invalid (>23)
            "day_of_week": 2
        }
    }

    response = client.post("/decide", json=request_data)

    # Should return 422 for validation error
    assert response.status_code == 422


def test_decide_endpoint_missing_field():
    """Test decide endpoint with missing required field."""
    request_data = {
        "user_features": {
            "user_id": 123
        },
        "candidate": {
            "template_id": "promo_001",
            # Missing channel
            "notification_type": "promo",
            "text": "Test",
            "hour_of_day": 14,
            "day_of_week": 2
        }
    }

    response = client.post("/decide", json=request_data)

    # Should return 422 for validation error
    assert response.status_code == 422


def test_batch_endpoint_empty_candidates():
    """Test batch endpoint with empty candidates list."""
    from tests.test_base_model import create_synthetic_dataset
    from src.api import server

    # Setup models
    df = create_synthetic_dataset(n_samples=500)
    click_model = BaseClickModel()
    click_model.train(df, target_col='clicked', verbose=False)
    complaint_model = BaseComplaintModel()
    complaint_model.train(df, target_col='unsubscribed', verbose=False)

    server.click_model = click_model
    server.complaint_model = complaint_model
    server.fatigue_model = FatigueModel()
    server.llm_client = MockLLMClient()
    server.decision_maker = NotificationDecisionMaker(
        click_model=click_model,
        complaint_model=complaint_model,
        fatigue_model=server.fatigue_model,
        llm_client=server.llm_client
    )

    request_data = {
        "user_features": {
            "user_id": 123,
            "notif_count_24h": 2,
            "notif_count_7d": 10,
            "notif_count_30d": 30
        },
        "candidates": []  # Empty
    }

    response = client.post("/decide_batch", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert len(data["decisions"]) == 0
    assert data["stats"]["total"] == 0


def test_train_endpoint_not_implemented():
    """Test that train endpoint returns 501."""
    response = client.post("/train?data_path=/tmp/data.parquet")
    assert response.status_code == 501


def test_health_memory_reporting():
    """Test that health endpoint reports reasonable memory usage."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    mem_mb = data["memory_mb"]

    # Memory should be positive and under 2GB (our constraint)
    assert 0 < mem_mb < 2048


def test_api_latency():
    """Test that API responses are reasonably fast."""
    import time
    from tests.test_base_model import create_synthetic_dataset
    from src.api import server

    # Setup models
    df = create_synthetic_dataset(n_samples=500)
    click_model = BaseClickModel()
    click_model.train(df, target_col='clicked', verbose=False)
    complaint_model = BaseComplaintModel()
    complaint_model.train(df, target_col='unsubscribed', verbose=False)

    server.click_model = click_model
    server.complaint_model = complaint_model
    server.fatigue_model = FatigueModel()
    server.llm_client = MockLLMClient()
    server.decision_maker = NotificationDecisionMaker(
        click_model=click_model,
        complaint_model=complaint_model,
        fatigue_model=server.fatigue_model,
        llm_client=server.llm_client
    )

    request_data = {
        "user_features": {
            "user_id": 123,
            "notif_count_24h": 2,
            "notif_count_7d": 10,
            "notif_count_30d": 30,
            "open_rate_7d": 0.15,
            "open_rate_30d": 0.12
        },
        "candidate": {
            "template_id": "promo_001",
            "channel": "push",
            "notification_type": "promo",
            "text": "Test",
            "hour_of_day": 14,
            "day_of_week": 2
        },
        "use_llm": False
    }

    # Measure latency
    start = time.time()
    response = client.post("/decide", json=request_data)
    latency_ms = (time.time() - start) * 1000

    assert response.status_code == 200

    # Should be under 500ms (our constraint)
    # Being generous with test environment
    assert latency_ms < 1000


def test_openapi_docs_available():
    """Test that OpenAPI documentation is available."""
    response = client.get("/docs")
    assert response.status_code == 200

    # Check that it's HTML (Swagger UI)
    assert "text/html" in response.headers.get("content-type", "")
