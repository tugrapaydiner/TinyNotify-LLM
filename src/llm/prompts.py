"""
LLM prompt templates for notification quality rating.
Designed for tiny language models (< 1B parameters) with minimal token usage.
"""
from typing import Dict, Optional
from src.features.schema import CandidateNotification, UserFeatures


def build_rating_prompt(
    candidate: CandidateNotification,
    max_tokens: int = 150
) -> str:
    """
    Build prompt for LLM to rate notification quality.

    Designed to be concise for tiny models with limited context windows.

    Args:
        candidate: Notification candidate to rate
        max_tokens: Maximum output tokens (for reference in prompt)

    Returns:
        Prompt string
    """
    # Extract user context
    user_features = candidate.user_features

    # Format notification type and channel
    notif_type = candidate.notification_type.value if hasattr(candidate.notification_type, 'value') else candidate.notification_type
    channel = candidate.channel.value if hasattr(candidate.channel, 'value') else candidate.channel

    # Build concise user context
    user_context = _build_user_context(user_features)

    # Build prompt
    prompt = f"""Rate this notification from 1-5 (1=poor, 5=excellent).

NOTIFICATION:
Type: {notif_type}
Channel: {channel}
Text: {candidate.text}

USER CONTEXT:
{user_context}

TASK:
Rate the notification quality (1-5) considering:
- Relevance to user
- Timing appropriateness
- Engagement likelihood
- Not being spammy

FORMAT:
Rating: [1-5]
Reason: [brief explanation in 1-2 sentences]
"""

    return prompt


def build_rating_prompt_minimal(
    candidate: CandidateNotification
) -> str:
    """
    Build minimal prompt for very tiny models.

    Uses absolute minimum tokens while maintaining clarity.

    Args:
        candidate: Notification candidate to rate

    Returns:
        Minimal prompt string
    """
    user_features = candidate.user_features

    notif_type = candidate.notification_type.value if hasattr(candidate.notification_type, 'value') else candidate.notification_type

    # Ultra-concise user context
    recent_notifs = user_features.notif_count_24h
    open_rate = int(user_features.open_rate_7d * 100)  # Convert to percentage

    prompt = f"""Rate notification 1-5.

Message: {candidate.text}
Type: {notif_type}

User:
- Received {recent_notifs} notifications today
- Opens {open_rate}% of notifications

Rating (1-5):"""

    return prompt


def build_rating_prompt_structured(
    candidate: CandidateNotification
) -> str:
    """
    Build structured prompt with clear sections.

    Good for models that work well with structured input.

    Args:
        candidate: Notification candidate to rate

    Returns:
        Structured prompt string
    """
    user_features = candidate.user_features

    notif_type = candidate.notification_type.value if hasattr(candidate.notification_type, 'value') else candidate.notification_type
    channel = candidate.channel.value if hasattr(candidate.channel, 'value') else candidate.channel

    # Hour bucket for timing context
    hour_bucket = user_features.most_active_hour_bucket or "unknown"
    current_hour = _get_hour_bucket(candidate.hour_of_day)

    prompt = f"""### TASK
Rate notification quality: 1 (poor) to 5 (excellent)

### NOTIFICATION
- Text: "{candidate.text}"
- Type: {notif_type}
- Channel: {channel}
- Timing: {current_hour}

### USER PROFILE
- Recent notifications: {user_features.notif_count_24h} today, {user_features.notif_count_7d} this week
- Engagement: {user_features.open_rate_7d:.1%} open rate
- Most active: {hour_bucket}
- Last notification: {_format_hours_since(user_features.hours_since_last_notification)}

### OUTPUT
Provide rating and brief reason:
Rating: [number 1-5]
Reason: [1 sentence]
"""

    return prompt


def _build_user_context(user_features: UserFeatures) -> str:
    """Build concise user context string."""
    lines = []

    # Volume context
    lines.append(f"Notifications: {user_features.notif_count_24h} today, {user_features.notif_count_7d} this week")

    # Engagement context
    if user_features.notif_count_7d >= 5:
        lines.append(f"Open rate: {user_features.open_rate_7d:.1%}")
    else:
        lines.append("Open rate: insufficient history")

    # Recency context
    if user_features.hours_since_last_notification is not None:
        hours_since = user_features.hours_since_last_notification
        if hours_since < 1:
            lines.append(f"Last notification: {int(hours_since * 60)} minutes ago")
        elif hours_since < 24:
            lines.append(f"Last notification: {hours_since:.1f} hours ago")
        else:
            lines.append(f"Last notification: {hours_since / 24:.1f} days ago")
    else:
        lines.append("Last notification: never")

    # Active time context
    if user_features.most_active_hour_bucket:
        lines.append(f"Most active: {user_features.most_active_hour_bucket}")

    return "\n".join(lines)


def _get_hour_bucket(hour: int) -> str:
    """Get hour bucket name."""
    if 0 <= hour < 6:
        return "night"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"


def _format_hours_since(hours: Optional[float]) -> str:
    """Format hours since last notification."""
    if hours is None:
        return "never"
    elif hours < 1:
        return f"{int(hours * 60)}min ago"
    elif hours < 24:
        return f"{hours:.1f}h ago"
    else:
        return f"{hours / 24:.1f}d ago"


def parse_llm_response(response: str, fallback_rating: int = 3) -> Dict[str, any]:
    """
    Parse LLM response to extract rating and reason.

    Args:
        response: Raw LLM response text
        fallback_rating: Rating to use if parsing fails

    Returns:
        Dict with 'rating' (int) and 'reason' (str)
    """
    result = {
        'rating': fallback_rating,
        'reason': 'Unable to parse LLM response'
    }

    # Try to extract rating
    rating = _extract_rating(response)
    if rating is not None:
        result['rating'] = rating

    # Try to extract reason
    reason = _extract_reason(response)
    if reason:
        result['reason'] = reason

    return result


def _extract_rating(response: str) -> Optional[int]:
    """Extract rating from LLM response."""
    import re

    # Look for patterns like "Rating: 4" or "Rating: [4]"
    patterns = [
        r'Rating:\s*\[?(\d)\]?',
        r'rating:\s*\[?(\d)\]?',
        r'Rate:\s*\[?(\d)\]?',
        r'rate:\s*\[?(\d)\]?',
        r'Score:\s*\[?(\d)\]?',
        r'score:\s*\[?(\d)\]?',
        r'^(\d)/5',  # Start of line with "4/5"
        r'\b([1-5])\b.*(?:out of|/)\s*5',  # "4 out of 5"
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            rating = int(match.group(1))
            if 1 <= rating <= 5:
                return rating

    # If no pattern matched, look for first digit 1-5 in response
    for char in response:
        if char in '12345':
            return int(char)

    return None


def _extract_reason(response: str) -> Optional[str]:
    """Extract reason from LLM response."""
    import re

    # Look for patterns like "Reason: text" or "Explanation: text"
    patterns = [
        r'Reason:\s*(.+?)(?:\n|$)',
        r'reason:\s*(.+?)(?:\n|$)',
        r'Explanation:\s*(.+?)(?:\n|$)',
        r'explanation:\s*(.+?)(?:\n|$)',
        r'Because:\s*(.+?)(?:\n|$)',
        r'because:\s*(.+?)(?:\n|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            reason = match.group(1).strip()
            # Take only first sentence if multiple
            reason = reason.split('.')[0] + '.' if '.' in reason else reason
            # Limit length
            if len(reason) > 200:
                reason = reason[:197] + '...'
            return reason

    # Fallback: take first line that's not the rating
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    for line in lines:
        # Skip lines that are just the rating
        if not re.match(r'^(Rating|Score|Rate):\s*\d$', line, re.IGNORECASE):
            if len(line) > 10:  # Must be substantial
                return line[:200]  # Limit length

    return None


# Prompt templates by strategy
PROMPT_STRATEGIES = {
    'default': build_rating_prompt,
    'minimal': build_rating_prompt_minimal,
    'structured': build_rating_prompt_structured,
}


def get_prompt(
    candidate: CandidateNotification,
    strategy: str = 'default',
    **kwargs
) -> str:
    """
    Get prompt using specified strategy.

    Args:
        candidate: Notification candidate
        strategy: Prompt strategy ('default', 'minimal', or 'structured')
        **kwargs: Additional arguments for prompt builder

    Returns:
        Prompt string
    """
    builder = PROMPT_STRATEGIES.get(strategy, build_rating_prompt)
    return builder(candidate, **kwargs)
