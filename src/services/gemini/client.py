import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from src.config import Settings
from src.exceptions import geminiConnectionError, geminiException, geminiTimeoutError
from src.schemas.gemini import RAGResponse
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)
class GeminiRAGClient:
    """Client for interacting with Gemini model for RAG."""

    def __init__(self, settings: Settings):
        """Initialize the Gemini RAG client.

        Args:
            settings: Application settings
        """
        self.api_url = settings.gemini_api_url
        self.timeout = settings.gemini_timeout
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    async def get_rag_response(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str,
    ) -> RAGResponse:
        """Get RAG response from Gemini model.

        Args:
            query: User's question
            chunks: Retrieved document chunks
            model: Gemini model to use

        Returns:
            RAGResponse object containing answer and sources
        """
        prompt = self.prompt_builder.create_rag_prompt(query, chunks)

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, json=payload)

            response.raise_for_status()
            response_data = response.json()

            answer, sources = self.response_parser.parse_response(
                response_data.get("text", ""), chunks
            )

            return RAGResponse(answer=answer, sources=sources)

        except httpx.TimeoutException as e:
            logger.error(f"Gemini request timed out: {e}")
            raise geminiTimeoutError("Request to Gemini model timed out.") from e
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during Gemini request: {e}")
            raise geminiConnectionError("Failed to connect to Gemini model.") from e
        except Exception as e:
            logger.error(f"Unexpected error during Gemini request: {e}")
            raise geminiException("An error occurred while processing the Gemini request.") from e
