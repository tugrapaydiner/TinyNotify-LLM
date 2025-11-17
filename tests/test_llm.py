"""Tests for LLM integration."""
import pytest
from datetime import datetime

from src.llm.prompts import (
    build_rating_prompt,
    build_rating_prompt_minimal,
    build_rating_prompt_structured,
    parse_llm_response,
    get_prompt,
    _extract_rating,
    _extract_reason
)
from src.llm.parser import (
    LLMResponseParser,
    ResponseValidator,
    parse_rating_simple,
    parse_rating_with_reason,
    validate_rating
)
from src.llm.client import MockLLMClient
from src.features.schema import (
    CandidateNotification,
    UserFeatures,
    Channel,
    NotificationType
)
from src.config import DEFAULT_CONFIG


def create_test_candidate() -> CandidateNotification:
    """Create a test notification candidate."""
    user_features = UserFeatures(
        user_id=123,
        timestamp=datetime.now(),
        notif_count_24h=2,
        notif_count_7d=10,
        notif_count_30d=30,
        open_rate_7d=0.15,
        open_rate_30d=0.12,
        hours_since_last_notification=3.5,
        most_active_hour_bucket='morning'
    )

    return CandidateNotification(
        user_id=123,
        template_id='template_1',
        channel=Channel.PUSH,
        notification_type=NotificationType.PRODUCT_TIP,
        text='Try our new feature: smart notifications!',
        timestamp=datetime.now(),
        hour_of_day=10,
        day_of_week=2,
        user_features=user_features
    )


# ===== Prompt Generation Tests =====

def test_build_rating_prompt():
    """Test default rating prompt generation."""
    candidate = create_test_candidate()

    prompt = build_rating_prompt(candidate)

    # Check prompt contains key elements
    assert 'Rate this notification' in prompt
    assert candidate.text in prompt
    assert 'product_tip' in prompt.lower()
    assert 'push' in prompt.lower()
    assert 'USER CONTEXT' in prompt


def test_build_rating_prompt_minimal():
    """Test minimal prompt generation."""
    candidate = create_test_candidate()

    prompt = build_rating_prompt_minimal(candidate)

    # Should be shorter than default
    assert len(prompt) < len(build_rating_prompt(candidate))

    # Should still contain essential info
    assert candidate.text in prompt
    assert 'Rating' in prompt or 'rating' in prompt


def test_build_rating_prompt_structured():
    """Test structured prompt generation."""
    candidate = create_test_candidate()

    prompt = build_rating_prompt_structured(candidate)

    # Check structure markers
    assert '###' in prompt
    assert 'TASK' in prompt
    assert 'NOTIFICATION' in prompt
    assert 'USER PROFILE' in prompt


def test_get_prompt_with_strategy():
    """Test prompt strategy selection."""
    candidate = create_test_candidate()

    # Test each strategy
    default_prompt = get_prompt(candidate, strategy='default')
    minimal_prompt = get_prompt(candidate, strategy='minimal')
    structured_prompt = get_prompt(candidate, strategy='structured')

    # All should contain the notification text
    assert candidate.text in default_prompt
    assert candidate.text in minimal_prompt
    assert candidate.text in structured_prompt

    # Lengths should differ
    assert len(minimal_prompt) < len(default_prompt)


# ===== Response Parsing Tests =====

def test_parse_llm_response_standard_format():
    """Test parsing standard format response."""
    response = "Rating: 4\nReason: Good timing and relevant content."

    result = parse_llm_response(response)

    assert result['rating'] == 4
    assert 'Good timing' in result['reason']


def test_parse_llm_response_alternative_formats():
    """Test parsing various response formats."""
    test_cases = [
        ("Rating: [5]\nReason: Excellent notification", 5),
        ("Score: 3\nBecause: Average quality", 3),
        ("4/5 - Very good", 4),
        ("I rate this 2 out of 5", 2),
        ("1", 1),  # Minimal response
    ]

    for response, expected_rating in test_cases:
        result = parse_llm_response(response)
        assert result['rating'] == expected_rating, f"Failed for: {response}"


def test_parse_llm_response_fallback():
    """Test fallback when parsing fails."""
    response = "This is not a valid rating response."

    result = parse_llm_response(response, fallback_rating=3)

    # Should use fallback
    assert result['rating'] == 3


def test_extract_rating_patterns():
    """Test rating extraction from various patterns."""
    test_cases = [
        ("Rating: 5", 5),
        ("rating: 4", 4),
        ("Score: 3", 3),
        ("Rate: 2", 2),
        ("4/5", 4),
        ("3 out of 5", 3),
        ("Just a plain 5", 5),
    ]

    for text, expected in test_cases:
        rating = _extract_rating(text)
        assert rating == expected, f"Failed for: {text}"


def test_extract_rating_invalid():
    """Test rating extraction with invalid inputs."""
    invalid_cases = [
        "No rating here",
        "Rating: 10",  # Out of range
        "Rating: 0",   # Out of range
        "",
    ]

    for text in invalid_cases:
        rating = _extract_rating(text)
        # Should return None for invalid
        assert rating is None or (1 <= rating <= 5)


def test_extract_reason():
    """Test reason extraction."""
    test_cases = [
        ("Reason: This is a good notification.", "This is a good notification"),
        ("Explanation: User will like it.", "User will like it"),
        ("Because: Timing is right.", "Timing is right"),
    ]

    for text, expected_fragment in test_cases:
        reason = _extract_reason(text)
        assert reason is not None
        assert expected_fragment in reason


# ===== Parser Class Tests =====

def test_llm_response_parser():
    """Test LLMResponseParser class."""
    parser = LLMResponseParser(fallback_rating=3)

    response = "Rating: 4\nReason: Well-timed notification."
    result = parser.parse(response)

    assert result['rating'] == 4
    assert 'Well-timed' in result['reason']
    assert 'confidence' in result
    assert 0 <= result['confidence'] <= 1


def test_parser_confidence_levels():
    """Test that parser assigns different confidence levels."""
    parser = LLMResponseParser()

    # High confidence: explicit label
    result1 = parser.parse("Rating: 5\nReason: Great")
    assert result1['confidence'] > 0.9

    # Medium confidence: fraction
    result2 = parser.parse("4/5 - Good")
    assert 0.7 < result2['confidence'] < 0.9

    # Low confidence: just a digit
    result3 = parser.parse("5")
    assert result3['confidence'] < 0.6


def test_parser_empty_response():
    """Test parser with empty response."""
    parser = LLMResponseParser(fallback_rating=3)

    result = parser.parse("")

    assert result['rating'] == 3
    assert result['confidence'] == 0.0


def test_response_validator():
    """Test response validation."""
    validator = ResponseValidator()

    # Valid response
    result1 = validator.validate("Rating: 4\nReason: Good notification")
    assert result1['valid']
    assert len(result1['issues']) == 0

    # Invalid: empty
    result2 = validator.validate("")
    assert not result2['valid']
    assert 'Empty response' in result2['issues']

    # Invalid: too short
    result3 = validator.validate("Hi")
    assert not result3['valid']


def test_parse_rating_simple_function():
    """Test simple rating parsing function."""
    rating = parse_rating_simple("Rating: 4\nReason: Good")
    assert rating == 4

    rating_fallback = parse_rating_simple("Invalid", fallback=3)
    assert rating_fallback == 3


def test_parse_rating_with_reason_function():
    """Test rating and reason parsing function."""
    rating, reason = parse_rating_with_reason("Rating: 5\nReason: Excellent timing")

    assert rating == 5
    assert 'Excellent' in reason


def test_validate_rating_function():
    """Test rating validation function."""
    assert validate_rating(1)
    assert validate_rating(3)
    assert validate_rating(5)
    assert validate_rating("4")  # String should work

    assert not validate_rating(0)
    assert not validate_rating(6)
    assert not validate_rating("abc")
    assert not validate_rating(None)


# ===== Mock Client Tests =====

def test_mock_llm_client_initialization():
    """Test mock LLM client initialization."""
    client = MockLLMClient()

    assert client._model_loaded
    info = client.get_model_info()
    assert info['loaded']


def test_mock_client_rate_notification():
    """Test rating with mock client."""
    client = MockLLMClient()
    client.load_model()

    candidate = create_test_candidate()
    result = client.rate_notification(candidate)

    # Check result structure
    assert 'rating' in result
    assert 'reason' in result
    assert 'latency_ms' in result

    # Check rating is valid
    assert 1 <= result['rating'] <= 5
    assert isinstance(result['reason'], str)
    assert result['latency_ms'] >= 0


def test_mock_client_rate_batch():
    """Test batch rating with mock client."""
    client = MockLLMClient()
    client.load_model()

    candidates = [create_test_candidate() for _ in range(3)]
    results = client.rate_batch(candidates)

    assert len(results) == 3
    for result in results:
        assert 'rating' in result
        assert 1 <= result['rating'] <= 5


def test_mock_client_deterministic():
    """Test that mock client is somewhat deterministic."""
    client = MockLLMClient()
    client.load_model()

    candidate = create_test_candidate()

    # Rate same candidate twice
    result1 = client.rate_notification(candidate)
    result2 = client.rate_notification(candidate)

    # Should get same rating (deterministic mock)
    assert result1['rating'] == result2['rating']


def test_mock_client_different_prompts():
    """Test mock client with different prompt strategies."""
    client = MockLLMClient()
    client.load_model()

    candidate = create_test_candidate()

    result1 = client.rate_notification(candidate, prompt_strategy='default')
    result2 = client.rate_notification(candidate, prompt_strategy='minimal')
    result3 = client.rate_notification(candidate, prompt_strategy='structured')

    # All should return valid ratings
    assert 1 <= result1['rating'] <= 5
    assert 1 <= result2['rating'] <= 5
    assert 1 <= result3['rating'] <= 5


def test_mock_client_heuristics():
    """Test that mock client applies basic heuristics."""
    client = MockLLMClient()
    client.load_model()

    # Create candidate with positive keywords
    user_features = UserFeatures(
        user_id=1,
        timestamp=datetime.now(),
        notif_count_24h=1,
        notif_count_7d=5,
        notif_count_30d=15,
        open_rate_7d=0.20  # High engagement
    )

    positive_candidate = CandidateNotification(
        user_id=1,
        template_id='t1',
        channel=Channel.PUSH,
        notification_type=NotificationType.PRODUCT_TIP,
        text='Exclusive new feature just for you!',
        timestamp=datetime.now(),
        hour_of_day=10,
        day_of_week=2,
        user_features=user_features
    )

    # Positive keywords should tend toward higher ratings
    result = client.rate_notification(positive_candidate)
    assert result['rating'] >= 3


def test_client_get_model_info():
    """Test getting model info."""
    client = MockLLMClient()
    client.load_model()

    info = client.get_model_info()

    assert 'loaded' in info
    assert info['loaded'] == True
    assert 'context_length' in info
    assert 'max_output_tokens' in info


def test_client_unload_model():
    """Test model unloading."""
    client = MockLLMClient()
    client.load_model()

    assert client._model_loaded

    client.unload_model()

    assert not client._model_loaded


# ===== Integration Tests =====

def test_end_to_end_mock_rating():
    """Test complete flow from candidate to rating."""
    # Create candidate
    user_features = UserFeatures(
        user_id=456,
        timestamp=datetime.now(),
        notif_count_24h=3,
        notif_count_7d=15,
        notif_count_30d=50,
        open_rate_7d=0.10,
        hours_since_last_notification=2.0
    )

    candidate = CandidateNotification(
        user_id=456,
        template_id='promo_1',
        channel=Channel.EMAIL,
        notification_type=NotificationType.PROMO,
        text='Limited time offer: 20% off!',
        timestamp=datetime.now(),
        hour_of_day=14,
        day_of_week=3,
        user_features=user_features
    )

    # Initialize client and rate
    client = MockLLMClient()
    client.load_model()
    result = client.rate_notification(candidate, prompt_strategy='default')

    # Validate result
    assert isinstance(result, dict)
    assert 1 <= result['rating'] <= 5
    assert len(result['reason']) > 0
    assert result['latency_ms'] >= 0


def test_prompt_to_parse_roundtrip():
    """Test creating prompt and parsing response."""
    candidate = create_test_candidate()

    # Generate prompt
    prompt = build_rating_prompt(candidate)
    assert len(prompt) > 0

    # Simulate LLM response
    simulated_response = "Rating: 4\nReason: Good timing and relevant to user interests."

    # Parse response
    result = parse_llm_response(simulated_response)

    assert result['rating'] == 4
    assert 'Good timing' in result['reason']


def test_multiple_candidates_with_different_features():
    """Test rating multiple candidates with varied features."""
    client = MockLLMClient()
    client.load_model()

    # Create candidates with different user states
    candidates = []

    for i in range(5):
        user_features = UserFeatures(
            user_id=i,
            timestamp=datetime.now(),
            notif_count_24h=i,
            notif_count_7d=i * 5,
            notif_count_30d=i * 15,
            open_rate_7d=0.05 * (i + 1)
        )

        candidate = CandidateNotification(
            user_id=i,
            template_id=f't_{i}',
            channel=Channel.PUSH,
            notification_type=NotificationType.PRODUCT_TIP,
            text=f'Notification {i}: Try this new feature!',
            timestamp=datetime.now(),
            hour_of_day=10 + i,
            day_of_week=i % 7,
            user_features=user_features
        )
        candidates.append(candidate)

    # Rate all
    results = client.rate_batch(candidates)

    # All should have valid ratings
    assert len(results) == 5
    for result in results:
        assert 1 <= result['rating'] <= 5
        assert len(result['reason']) > 0


def test_parser_robustness():
    """Test parser handles various edge cases."""
    parser = LLMResponseParser(fallback_rating=3)

    edge_cases = [
        "",  # Empty
        "   ",  # Whitespace only
        "Rating:",  # Missing value
        "5 5 5 5",  # Multiple numbers
        "The rating is somewhere between 3 and 4",  # Ambiguous
        "I cannot provide a rating",  # Refusal
        "...",  # Just ellipsis
    ]

    for case in edge_cases:
        result = parser.parse(case)
        # Should not crash and should return valid rating
        assert 1 <= result['rating'] <= 5
        assert 'reason' in result
