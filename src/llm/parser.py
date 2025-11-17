"""
LLM response parsing utilities.
Robust extraction of ratings and reasons from varied LLM outputs.
"""
import re
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class LLMResponseParser:
    """
    Parser for LLM responses.

    Handles various response formats from different tiny LLMs.
    """

    def __init__(self, fallback_rating: int = 3):
        """
        Initialize parser.

        Args:
            fallback_rating: Rating to use if parsing fails
        """
        self.fallback_rating = fallback_rating

    def parse(self, response: str) -> Dict[str, any]:
        """
        Parse LLM response to extract rating and reason.

        Args:
            response: Raw LLM response text

        Returns:
            Dict with 'rating' (int), 'reason' (str), and 'confidence' (float)
        """
        if not response or not response.strip():
            return {
                'rating': self.fallback_rating,
                'reason': 'Empty response',
                'confidence': 0.0
            }

        # Extract rating
        rating, rating_confidence = self._extract_rating_with_confidence(response)

        # Extract reason
        reason = self._extract_reason(response)

        return {
            'rating': rating if rating is not None else self.fallback_rating,
            'reason': reason or 'No reason provided',
            'confidence': rating_confidence
        }

    def _extract_rating_with_confidence(
        self,
        response: str
    ) -> Tuple[Optional[int], float]:
        """
        Extract rating with confidence score.

        Args:
            response: Response text

        Returns:
            Tuple of (rating, confidence)
        """
        # High confidence patterns (explicit labels)
        high_conf_patterns = [
            (r'Rating:\s*\[?(\d)\]?', 0.95),
            (r'rating:\s*\[?(\d)\]?', 0.95),
            (r'Score:\s*\[?(\d)\]?', 0.90),
            (r'score:\s*\[?(\d)\]?', 0.90),
        ]

        for pattern, confidence in high_conf_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                rating = int(match.group(1))
                if 1 <= rating <= 5:
                    return rating, confidence

        # Medium confidence patterns
        med_conf_patterns = [
            (r'^(\d)/5', 0.80),  # "4/5"
            (r'\b([1-5])\s*out of\s*5', 0.75),  # "4 out of 5"
            (r'\b([1-5])\s*/\s*5', 0.75),  # "4 / 5"
        ]

        for pattern, confidence in med_conf_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                rating = int(match.group(1))
                return rating, confidence

        # Low confidence: first digit 1-5 in response
        for char in response:
            if char in '12345':
                return int(char), 0.50

        return None, 0.0

    def _extract_reason(self, response: str) -> Optional[str]:
        """
        Extract reason from response.

        Args:
            response: Response text

        Returns:
            Extracted reason or None
        """
        # Labeled reason patterns
        patterns = [
            r'Reason:\s*(.+?)(?:\n\n|\n$|$)',
            r'reason:\s*(.+?)(?:\n\n|\n$|$)',
            r'Explanation:\s*(.+?)(?:\n\n|\n$|$)',
            r'explanation:\s*(.+?)(?:\n\n|\n$|$)',
            r'Because:\s*(.+?)(?:\n\n|\n$|$)',
            r'because:\s*(.+?)(?:\n\n|\n$|$)',
            r'Justification:\s*(.+?)(?:\n\n|\n$|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                reason = self._clean_reason(match.group(1))
                if reason:
                    return reason

        # Fallback: extract substantive lines
        reason = self._extract_fallback_reason(response)
        if reason:
            return reason

        return None

    def _clean_reason(self, text: str) -> str:
        """
        Clean and validate reason text.

        Args:
            text: Raw reason text

        Returns:
            Cleaned reason
        """
        # Strip whitespace
        text = text.strip()

        # Take only first sentence if multiple
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            text = sentences[0].strip()
            # Add period if not present
            if text and text[-1] not in '.!?':
                text += '.'

        # Limit length
        max_length = 200
        if len(text) > max_length:
            text = text[:max_length - 3] + '...'

        # Validate it's substantive (at least 10 chars, contains some letters)
        if len(text) < 10 or not re.search(r'[a-zA-Z]', text):
            return ""

        return text

    def _extract_fallback_reason(self, response: str) -> Optional[str]:
        """
        Extract reason as fallback when no labeled reason found.

        Args:
            response: Full response

        Returns:
            Extracted reason or None
        """
        # Split into lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        # Skip lines that are just the rating
        rating_patterns = [
            r'^(Rating|Score|Rate):\s*\d$',
            r'^\d/5$',
            r'^\d\s*out of\s*5$',
        ]

        for line in lines:
            # Skip rating lines
            is_rating = any(re.match(p, line, re.IGNORECASE) for p in rating_patterns)
            if is_rating:
                continue

            # Check if line is substantive
            if len(line) > 15 and re.search(r'[a-zA-Z]', line):
                return self._clean_reason(line)

        return None


class ResponseValidator:
    """
    Validator for LLM response quality.

    Checks if response meets quality standards.
    """

    def __init__(self):
        """Initialize validator."""
        pass

    def validate(self, response: str) -> Dict[str, any]:
        """
        Validate response quality.

        Args:
            response: Raw response text

        Returns:
            Dict with validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        # Check for empty response
        if not response or not response.strip():
            results['valid'] = False
            results['issues'].append('Empty response')
            return results

        # Check minimum length
        if len(response.strip()) < 5:
            results['valid'] = False
            results['issues'].append('Response too short')

        # Check if contains a number (likely rating)
        if not re.search(r'\d', response):
            results['warnings'].append('No numeric rating found')

        # Check if response is cut off (ends mid-word)
        if len(response) > 10 and not response.rstrip()[-1] in '.!?\n':
            results['warnings'].append('Response may be incomplete')

        # Check for common error patterns
        error_patterns = [
            (r'error', 'Contains error message'),
            (r'failed', 'Contains failure message'),
            (r'unable', 'Indicates inability to complete'),
        ]

        for pattern, message in error_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                results['warnings'].append(message)

        return results


def parse_rating_simple(response: str, fallback: int = 3) -> int:
    """
    Simple function to extract just the rating.

    Args:
        response: LLM response
        fallback: Fallback rating

    Returns:
        Rating from 1-5
    """
    parser = LLMResponseParser(fallback)
    result = parser.parse(response)
    return result['rating']


def parse_rating_with_reason(
    response: str,
    fallback: int = 3
) -> Tuple[int, str]:
    """
    Extract rating and reason.

    Args:
        response: LLM response
        fallback: Fallback rating

    Returns:
        Tuple of (rating, reason)
    """
    parser = LLMResponseParser(fallback)
    result = parser.parse(response)
    return result['rating'], result['reason']


def validate_rating(rating: any) -> bool:
    """
    Validate that rating is in valid range.

    Args:
        rating: Rating value

    Returns:
        True if valid, False otherwise
    """
    try:
        rating_int = int(rating)
        return 1 <= rating_int <= 5
    except (ValueError, TypeError):
        return False
