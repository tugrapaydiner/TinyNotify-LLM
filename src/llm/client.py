"""
LLM client for notification quality rating.
Uses tiny local models (< 1B parameters) for CPU-only inference.
"""
import time
from typing import Optional, Dict, List
from pathlib import Path
import logging

from src.config import Config, DEFAULT_CONFIG
from src.features.schema import CandidateNotification
from src.llm.prompts import get_prompt, parse_llm_response

logger = logging.getLogger(__name__)


class TinyLLMClient:
    """
    Client for tiny local LLM inference.

    Designed for CPU-only operation with minimal memory footprint.
    Supports llama-cpp-python backend for GGUF models.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize LLM client.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config if config is not None else DEFAULT_CONFIG
        self.model = None
        self._model_loaded = False

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load LLM model from disk.

        Args:
            model_path: Path to model file (uses config if None)
        """
        if model_path is None:
            model_path = self.config.llm.model_path

        if model_path is None:
            logger.warning("No model path provided. LLM will use fallback ratings.")
            return

        model_path = Path(model_path)

        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. LLM will use fallback ratings.")
            return

        try:
            from llama_cpp import Llama

            logger.info(f"Loading LLM model from {model_path}...")

            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.llm.max_context_length,
                n_threads=self.config.llm.n_threads,
                n_batch=self.config.llm.n_batch,
                use_mmap=True,  # Memory-map the model for efficiency
                use_mlock=False,  # Don't lock memory (allows swapping)
                verbose=False
            )

            self._model_loaded = True
            logger.info("LLM model loaded successfully")

        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            logger.warning("LLM will use fallback ratings.")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            logger.warning("LLM will use fallback ratings.")

    def rate_notification(
        self,
        candidate: CandidateNotification,
        prompt_strategy: str = 'default',
        timeout: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Rate a notification using the LLM.

        Args:
            candidate: Notification to rate
            prompt_strategy: Prompt strategy ('default', 'minimal', 'structured')
            timeout: Timeout in seconds (uses config if None)

        Returns:
            Dict with 'rating' (int 1-5), 'reason' (str), and 'latency_ms' (float)
        """
        if timeout is None:
            timeout = self.config.llm.timeout_seconds

        start_time = time.time()

        # If model not loaded, use fallback
        if not self._model_loaded or self.model is None:
            return {
                'rating': self.config.llm.fallback_rating,
                'reason': 'LLM not available (using fallback)',
                'latency_ms': 0.0
            }

        # Build prompt
        prompt = get_prompt(candidate, strategy=prompt_strategy)

        # Generate response with retry logic
        response_text = None
        for attempt in range(self.config.llm.max_retries + 1):
            try:
                response_text = self._generate(prompt, timeout)
                break
            except TimeoutError:
                logger.warning(f"LLM generation timeout (attempt {attempt + 1})")
                if attempt == self.config.llm.max_retries:
                    logger.error("All LLM retry attempts failed")
                    return {
                        'rating': self.config.llm.fallback_rating,
                        'reason': 'LLM timeout (using fallback)',
                        'latency_ms': (time.time() - start_time) * 1000
                    }
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                return {
                    'rating': self.config.llm.fallback_rating,
                    'reason': f'LLM error (using fallback): {str(e)[:50]}',
                    'latency_ms': (time.time() - start_time) * 1000
                }

        # Parse response
        if response_text is None:
            result = {
                'rating': self.config.llm.fallback_rating,
                'reason': 'No response from LLM (using fallback)'
            }
        else:
            result = parse_llm_response(response_text, self.config.llm.fallback_rating)

        # Add latency
        result['latency_ms'] = (time.time() - start_time) * 1000

        return result

    def _generate(self, prompt: str, timeout: float) -> str:
        """
        Generate text from LLM with timeout.

        Args:
            prompt: Input prompt
            timeout: Timeout in seconds

        Returns:
            Generated text

        Raises:
            TimeoutError: If generation exceeds timeout
        """
        import signal

        # Define timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError("LLM generation exceeded timeout")

        # Set alarm for timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout) + 1)  # Add 1 second buffer

        try:
            output = self.model(
                prompt,
                max_tokens=self.config.llm.max_tokens_output,
                temperature=self.config.llm.temperature,
                top_p=self.config.llm.top_p,
                stop=["###", "\n\n\n"],  # Stop tokens
                echo=False
            )

            # Cancel alarm
            signal.alarm(0)

            # Extract text from output
            if isinstance(output, dict) and 'choices' in output:
                return output['choices'][0]['text'].strip()
            else:
                return str(output).strip()

        except TimeoutError:
            signal.alarm(0)
            raise
        except Exception as e:
            signal.alarm(0)
            raise e

    def rate_batch(
        self,
        candidates: List[CandidateNotification],
        prompt_strategy: str = 'default'
    ) -> List[Dict[str, any]]:
        """
        Rate multiple notifications.

        Note: Currently processes sequentially. Batch processing can be
        added for efficiency if needed.

        Args:
            candidates: List of notifications to rate
            prompt_strategy: Prompt strategy to use

        Returns:
            List of rating results
        """
        results = []

        for candidate in candidates:
            result = self.rate_notification(candidate, prompt_strategy)
            results.append(result)

        return results

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about loaded model.

        Returns:
            Dict with model information
        """
        info = {
            'loaded': self._model_loaded,
            'model_path': str(self.config.llm.model_path) if self.config.llm.model_path else None,
            'context_length': self.config.llm.max_context_length,
            'max_output_tokens': self.config.llm.max_tokens_output,
        }

        if self._model_loaded and self.model is not None:
            try:
                # Try to get model metadata
                info['n_ctx'] = self.model.n_ctx()
                info['n_vocab'] = self.model.n_vocab()
            except:
                pass

        return info

    def unload_model(self) -> None:
        """
        Unload model from memory.

        Useful for freeing up resources when LLM is not needed.
        """
        if self.model is not None:
            del self.model
            self.model = None
            self._model_loaded = False
            logger.info("LLM model unloaded")


class MockLLMClient(TinyLLMClient):
    """
    Mock LLM client for testing.

    Returns deterministic ratings based on simple heuristics.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize mock client."""
        super().__init__(config)
        self._model_loaded = True  # Pretend model is loaded

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Mock model loading (does nothing)."""
        self._model_loaded = True
        logger.info("Mock LLM client initialized (no actual model loaded)")

    def _generate(self, prompt: str, timeout: float) -> str:
        """
        Generate mock response based on simple heuristics.

        Args:
            prompt: Input prompt
            timeout: Timeout (ignored)

        Returns:
            Mock response text
        """
        # Extract notification text from prompt
        notif_text = self._extract_notification_text(prompt)

        # Simple heuristic: rate based on text length and keywords
        rating = self._compute_mock_rating(notif_text, prompt)

        # Generate mock response
        response = f"Rating: {rating}\nReason: Mock rating based on text analysis."

        return response

    def _extract_notification_text(self, prompt: str) -> str:
        """Extract notification text from prompt."""
        # Look for text after "Text:" or "Message:"
        import re

        patterns = [
            r'Text:\s*(.+?)(?:\n|$)',
            r'Message:\s*(.+?)(?:\n|$)',
            r'"(.+?)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    def _compute_mock_rating(self, notif_text: str, full_prompt: str) -> int:
        """
        Compute mock rating using simple heuristics.

        Args:
            notif_text: Notification text
            full_prompt: Full prompt (for context)

        Returns:
            Rating from 1-5
        """
        rating = 3  # Default

        # Positive keywords increase rating
        positive_keywords = ['exclusive', 'special', 'new', 'save', 'limited', 'tip']
        negative_keywords = ['spam', 'urgent', 'click now', 'act now', 'last chance']

        text_lower = notif_text.lower()

        for keyword in positive_keywords:
            if keyword in text_lower:
                rating = min(5, rating + 1)
                break

        for keyword in negative_keywords:
            if keyword in text_lower:
                rating = max(1, rating - 1)
                break

        # Check user engagement from prompt
        if 'Opens 0%' in full_prompt or 'Opens 1%' in full_prompt:
            rating = max(1, rating - 1)  # Low engagement suggests poor notifications
        elif 'Opens 2' in full_prompt or 'Opens 3' in full_prompt:
            rating = min(5, rating + 1)  # High engagement suggests good notifications

        # Check notification volume
        if 'Received 10' in full_prompt or 'Received 1' in full_prompt:
            rating = max(1, rating - 1)  # Too many notifications

        return rating

    def unload_model(self) -> None:
        """Unload mock model (sets loaded flag to False)."""
        self._model_loaded = False
        logger.info("Mock LLM client unloaded")
