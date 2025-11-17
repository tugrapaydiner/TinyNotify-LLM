"""Tests for policy and scoring logic."""
import pytest
import pandas as pd
from datetime import datetime

from src.policy.scoring import NotificationScorer, AdaptiveScorer
from src.policy.decision import NotificationDecisionMaker
from src.features.schema import (
    CandidateNotification,
    UserFeatures,
    Channel,
    NotificationType
)
from src.models.base_model import BaseClickModel, BaseComplaintModel
from src.models.fatigue_model import FatigueModel
from src.llm.client import MockLLMClient
from src.config import DEFAULT_CONFIG

# Import test utilities from other test files
import sys
sys.path.insert(0, 'tests')
from test_base_model import create_synthetic_dataset


def create_test_candidate(
    user_id: int = 1,
    notif_count_24h: int = 2,
    notif_count_7d: int = 10,
    open_rate_7d: float = 0.15
) -> CandidateNotification:
    """Create a test candidate notification."""
    user_features = UserFeatures(
        user_id=user_id,
        timestamp=datetime.now(),
        notif_count_24h=notif_count_24h,
        notif_count_7d=notif_count_7d,
        notif_count_30d=max(notif_count_7d, 30),
        open_rate_7d=open_rate_7d,
        open_rate_30d=open_rate_7d * 0.9,
        hours_since_last_notification=3.0
    )

    return CandidateNotification(
        user_id=user_id,
        template_id='template_1',
        channel=Channel.PUSH,
        notification_type=NotificationType.PRODUCT_TIP,
        text='Check out this new feature!',
        timestamp=datetime.now(),
        hour_of_day=10,
        day_of_week=2,
        user_features=user_features
    )


# ===== Scorer Tests =====

def test_scorer_initialization():
    """Test scorer initialization."""
    scorer = NotificationScorer()
    assert scorer.config is not None


def test_compute_base_score():
    """Test base score computation."""
    scorer = NotificationScorer()

    # High click, low complaint -> positive score
    score1 = scorer.compute_base_score(p_click=0.3, p_complaint=0.01)
    assert score1 > 0

    # Low click, high complaint -> negative score
    score2 = scorer.compute_base_score(p_click=0.05, p_complaint=0.1)
    assert score2 < 0

    # Equal -> depends on lambda
    score3 = scorer.compute_base_score(p_click=0.1, p_complaint=0.02)
    # With lambda=5.0: 0.1 - 5*0.02 = 0.0
    assert abs(score3) < 0.001


def test_normalize_llm_rating():
    """Test LLM rating normalization."""
    scorer = NotificationScorer()

    assert scorer.normalize_llm_rating(1) == 0.0  # Min
    assert scorer.normalize_llm_rating(5) == 1.0  # Max
    assert scorer.normalize_llm_rating(3) == 0.5  # Middle

    # Out of range handling
    assert scorer.normalize_llm_rating(0) == 0.0  # Clips to 1
    assert scorer.normalize_llm_rating(6) == 1.0  # Clips to 5


def test_compute_final_score():
    """Test final score computation."""
    scorer = NotificationScorer()

    score = scorer.compute_final_score(
        p_click=0.3,
        p_complaint=0.01,
        llm_rating=4,
        fatigue_penalty=1.0
    )

    # Score should be positive
    assert score > 0

    # Score with fatigue should be lower
    score_with_fatigue = scorer.compute_final_score(
        p_click=0.3,
        p_complaint=0.01,
        llm_rating=4,
        fatigue_penalty=0.5
    )

    assert score_with_fatigue < score


def test_score_candidate():
    """Test scoring a candidate."""
    scorer = NotificationScorer()
    candidate = create_test_candidate()

    scored = scorer.score_candidate(
        candidate,
        p_click=0.3,
        p_complaint=0.02,
        llm_rating=4,
        fatigue_penalty=1.0
    )

    assert scored.p_click == 0.3
    assert scored.p_complaint == 0.02
    assert scored.llm_rating == 4
    assert scored.base_score is not None
    assert scored.final_score is not None


def test_score_batch():
    """Test batch scoring."""
    scorer = NotificationScorer()

    candidates = [create_test_candidate(user_id=i) for i in range(5)]
    p_clicks = [0.2, 0.3, 0.15, 0.25, 0.1]
    p_complaints = [0.01, 0.02, 0.01, 0.03, 0.05]

    scored_list = scorer.score_batch(candidates, p_clicks, p_complaints)

    assert len(scored_list) == 5
    for scored in scored_list:
        assert scored.final_score is not None


def test_rank_candidates():
    """Test ranking scored candidates."""
    scorer = NotificationScorer()

    candidates = [create_test_candidate(user_id=i) for i in range(3)]
    p_clicks = [0.1, 0.3, 0.2]  # Second should rank highest
    p_complaints = [0.01, 0.01, 0.01]

    scored_list = scorer.score_batch(candidates, p_clicks, p_complaints)
    ranked = scorer.rank_candidates(scored_list)

    # Highest score should be first
    assert ranked[0].p_click == 0.3


def test_get_top_k():
    """Test getting top K candidates."""
    scorer = NotificationScorer()

    candidates = [create_test_candidate(user_id=i) for i in range(10)]
    p_clicks = [0.1 + i * 0.02 for i in range(10)]  # Increasing
    p_complaints = [0.01] * 10

    scored_list = scorer.score_batch(candidates, p_clicks, p_complaints)
    top_3 = scorer.get_top_k(scored_list, k=3)

    assert len(top_3) == 3
    # Top should have highest scores
    assert top_3[0].final_score >= top_3[1].final_score
    assert top_3[1].final_score >= top_3[2].final_score


def test_compute_score_statistics():
    """Test score statistics computation."""
    scorer = NotificationScorer()

    candidates = [create_test_candidate(user_id=i) for i in range(5)]
    p_clicks = [0.2] * 5
    p_complaints = [0.01] * 5

    scored_list = scorer.score_batch(candidates, p_clicks, p_complaints)
    stats = scorer.compute_score_statistics(scored_list)

    assert stats['count'] == 5
    assert 'mean_score' in stats
    assert 'median_score' in stats
    assert 'min_score' in stats
    assert 'max_score' in stats


def test_score_statistics_empty():
    """Test statistics with empty list."""
    scorer = NotificationScorer()

    stats = scorer.compute_score_statistics([])

    assert stats['count'] == 0
    assert stats['mean_score'] == 0.0


# ===== Adaptive Scorer Tests =====

def test_adaptive_scorer_initialization():
    """Test adaptive scorer initialization."""
    scorer = AdaptiveScorer()

    assert scorer.alpha > 0
    assert scorer.beta > 0


def test_adaptive_scorer_update_weights():
    """Test updating adaptive weights."""
    scorer = AdaptiveScorer()

    initial_alpha = scorer.alpha
    initial_beta = scorer.beta

    # Update with LLM being more accurate
    scorer.update_weights(base_model_accuracy=0.6, llm_accuracy=0.8)

    # Beta should increase (LLM more accurate)
    assert scorer.beta > initial_beta
    assert scorer.alpha < initial_alpha

    # Weights should sum to ~1
    assert abs((scorer.alpha + scorer.beta) - 1.0) < 0.001


def test_adaptive_scorer_get_weights():
    """Test getting current weights."""
    scorer = AdaptiveScorer()

    weights = scorer.get_current_weights()

    assert 'alpha' in weights
    assert 'beta' in weights
    assert weights['alpha'] > 0
    assert weights['beta'] > 0


# ===== Decision Maker Tests =====

def create_test_decision_maker():
    """Create a decision maker with trained models."""
    # Create and train models
    df = create_synthetic_dataset(n_samples=500)

    click_model = BaseClickModel()
    click_model.train(df, target_col='clicked', verbose=False)

    complaint_model = BaseComplaintModel()
    complaint_model.train(df, target_col='unsubscribed', verbose=False)

    fatigue_model = FatigueModel()

    llm_client = MockLLMClient()
    llm_client.load_model()

    decision_maker = NotificationDecisionMaker(
        click_model=click_model,
        complaint_model=complaint_model,
        fatigue_model=fatigue_model,
        llm_client=llm_client
    )

    return decision_maker


def test_decision_maker_initialization():
    """Test decision maker initialization."""
    dm = create_test_decision_maker()

    assert dm.click_model is not None
    assert dm.complaint_model is not None
    assert dm.fatigue_model is not None
    assert dm.scorer is not None


def test_decide_single():
    """Test single decision."""
    dm = create_test_decision_maker()
    candidate = create_test_candidate()

    decision = dm.decide_single(candidate, use_llm=True)

    assert decision is not None
    assert decision.candidate == candidate
    assert decision.scored is not None
    assert isinstance(decision.should_send, bool)
    assert decision.reason is not None
    assert decision.decision_timestamp is not None


def test_decide_single_without_llm():
    """Test single decision without LLM."""
    dm = create_test_decision_maker()
    candidate = create_test_candidate()

    decision = dm.decide_single(candidate, use_llm=False)

    assert decision is not None
    assert isinstance(decision.should_send, bool)


def test_decide_batch():
    """Test batch decision making."""
    dm = create_test_decision_maker()

    candidates = [create_test_candidate(user_id=i) for i in range(10)]

    decisions = dm.decide_batch(candidates, use_llm_for_top_k=True, top_k=3)

    assert len(decisions) == 10
    for decision in decisions:
        assert isinstance(decision.should_send, bool)
        assert decision.reason is not None


def test_decide_batch_without_llm():
    """Test batch decisions without LLM."""
    dm = create_test_decision_maker()

    candidates = [create_test_candidate(user_id=i) for i in range(5)]

    decisions = dm.decide_batch(candidates, use_llm_for_top_k=False)

    assert len(decisions) == 5


def test_decision_thresholding():
    """Test that decisions respect score threshold."""
    dm = create_test_decision_maker()

    # Create candidates with varying characteristics
    good_candidate = create_test_candidate(
        notif_count_24h=1,
        notif_count_7d=5,
        open_rate_7d=0.30  # High engagement
    )

    poor_candidate = create_test_candidate(
        notif_count_24h=10,
        notif_count_7d=50,
        open_rate_7d=0.01  # Low engagement, high volume
    )

    decision_good = dm.decide_single(good_candidate)
    decision_poor = dm.decide_single(poor_candidate)

    # Good candidate more likely to be sent
    if decision_good.should_send and not decision_poor.should_send:
        assert decision_good.scored.final_score > decision_poor.scored.final_score


def test_decision_stats():
    """Test decision statistics computation."""
    dm = create_test_decision_maker()

    candidates = [create_test_candidate(user_id=i) for i in range(20)]
    decisions = dm.decide_batch(candidates)

    stats = dm.get_decision_stats(decisions)

    assert stats['total'] == 20
    assert stats['sent'] + stats['rejected'] == 20
    assert 0 <= stats['send_rate'] <= 1
    assert 'mean_score' in stats


def test_decision_stats_empty():
    """Test statistics with empty decisions."""
    dm = create_test_decision_maker()

    stats = dm.get_decision_stats([])

    assert stats['total'] == 0
    assert stats['sent'] == 0


def test_feature_extraction():
    """Test feature extraction from candidate."""
    dm = create_test_decision_maker()
    candidate = create_test_candidate()

    features = dm._extract_features(candidate)

    # Check required features exist
    assert 'notif_count_24h' in features
    assert 'open_rate_168h' in features
    assert 'channel_encoded' in features
    assert 'hour_of_day' in features
    assert 'is_weekend' in features


def test_batch_efficiency():
    """Test that batch processing is more efficient than single."""
    import time

    dm = create_test_decision_maker()

    candidates = [create_test_candidate(user_id=i) for i in range(20)]

    # Batch processing
    start = time.time()
    dm.decide_batch(candidates, use_llm_for_top_k=False)
    batch_time = time.time() - start

    # Single processing
    start = time.time()
    for candidate in candidates:
        dm.decide_single(candidate, use_llm=False)
    single_time = time.time() - start

    # Batch should be faster (or at least not much slower)
    # Being lenient with timing requirements
    assert batch_time < single_time * 2


def test_decision_reasons():
    """Test that decision reasons are generated."""
    dm = create_test_decision_maker()

    candidates = [
        create_test_candidate(notif_count_24h=1, notif_count_7d=5, open_rate_7d=0.3),  # Good
        create_test_candidate(notif_count_24h=8, notif_count_7d=40, open_rate_7d=0.01),  # Bad
    ]

    decisions = dm.decide_batch(candidates)

    for decision in decisions:
        assert decision.reason is not None
        assert len(decision.reason) > 10  # Substantive reason


def test_integration_end_to_end():
    """Test complete flow from candidate to decision."""
    # Create decision maker
    dm = create_test_decision_maker()

    # Create diverse candidates
    candidates = []

    # Good candidate
    candidates.append(create_test_candidate(
        user_id=1,
        notif_count_24h=2,
        notif_count_7d=10,
        open_rate_7d=0.25
    ))

    # Moderate candidate
    candidates.append(create_test_candidate(
        user_id=2,
        notif_count_24h=4,
        notif_count_7d=20,
        open_rate_7d=0.10
    ))

    # Poor candidate (fatigued)
    candidates.append(create_test_candidate(
        user_id=3,
        notif_count_24h=12,
        notif_count_7d=60,
        open_rate_7d=0.02
    ))

    # Make decisions
    decisions = dm.decide_batch(candidates, use_llm_for_top_k=True, top_k=2)

    # Validate results
    assert len(decisions) == 3

    for decision in decisions:
        # All should have valid structure
        assert decision.candidate is not None
        assert decision.scored is not None
        assert isinstance(decision.should_send, bool)
        assert decision.reason is not None

        # Scores should be in reasonable range
        assert -1 <= decision.scored.final_score <= 2


def test_scorer_with_zero_fatigue():
    """Test scoring with zero fatigue penalty."""
    scorer = NotificationScorer()
    candidate = create_test_candidate()

    scored = scorer.score_candidate(
        candidate,
        p_click=0.3,
        p_complaint=0.01,
        llm_rating=5,
        fatigue_penalty=0.0  # Complete suppression
    )

    assert scored.final_score == 0.0


def test_scorer_with_varying_llm_ratings():
    """Test that LLM ratings affect scores appropriately."""
    scorer = NotificationScorer()
    candidate = create_test_candidate()

    p_click = 0.2
    p_complaint = 0.01

    scores = []
    for rating in [1, 2, 3, 4, 5]:
        scored = scorer.score_candidate(
            candidate, p_click, p_complaint, rating, fatigue_penalty=1.0
        )
        scores.append(scored.final_score)

    # Scores should increase with rating
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1]
