import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.schemas.ollama import RAGResponse
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser


logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google Gemini API - tương thích với OllamaClient."""

    def __init__(self, settings: Settings):
        api_key = settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        # SDK mới: tạo client với api_key
        self.client = genai.Client(api_key=api_key)

        self.default_model = settings.gemini_model or "gemini-1.5-flash"
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()  # Nếu bạn dùng structured parsing

        # Cấu hình generation mặc định
        self.generation_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=2048,
        )

        logger.info(f"GeminiClient initialized with model: {self.default_model}")

    async def health_check(self) -> Dict[str, Any]:
        """Check if Gemini service is healthy."""
        try:
            response = self.client.models.generate_content(
                model=self.default_model,
                contents="Respond with only the word: OK",
                generation_config=types.GenerateContentConfig(max_output_tokens=5),
            )
            if response.text and "OK" in response.text.strip().upper():
                return {
                    "status": "healthy",
                    "message": "Gemini service is running",
                    "model": self.default_model,
                }
            else:
                raise GeminiException("Health check failed: unexpected response")
        except genai.types.BlockedPromptException as e:
            raise GeminiException(f"Content blocked: {e}")
        except genai.types.RateLimitError as e:
            raise GeminiException(f"Rate limit exceeded: {e}")
        except Exception as e:
            raise GeminiConnectionError(f"Gemini health check failed: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Gemini models that support generateContent."""
        try:
            models = []
            for m in self.client.models.list():
                if "generateContent" in m.supported_generation_methods:
                    models.append({
                        "name": m.name.split("/")[-1],  # e.g., "gemini-1.5-pro"
                        "display_name": getattr(m, "display_name", m.name),
                        "description": getattr(m, "description", ""),
                    })
            return models
        except Exception as e:
            raise GeminiException(f"Failed to list models: {e}")

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate text (non-streaming) - tương thích với Ollama.generate"""
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                generation_config=types.GenerateContentConfig(**kwargs),
                stream=stream,
            )
            if stream:
                return response  # Trả về iterator cho stream
            return {"response": response.text}
        except Exception as e:
            raise GeminiException(f"Generation failed: {e}")

    async def generate_stream(self, model: str, prompt: str, **kwargs):
        """Streaming generation - tương thích với Ollama.generate_stream"""
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                generation_config=types.GenerateContentConfig(**kwargs),
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield {"response": chunk.text}
        except Exception as e:
            raise GeminiException(f"Streaming generation failed: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str | None = None,
        use_structured_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate RAG answer - tương thích 100% với OllamaClient.generate_rag_answer
        """
        try:
            model_name = model or self.default_model
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            logger.info(f"Generating RAG answer with Gemini model: {model_name}")

            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                generation_config=self.generation_config,
            )

            # Kiểm tra safety blocking
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                raise GeminiException(f"Content blocked: {response.prompt_feedback.block_reason}")

            raw_answer = response.text.strip() if response.text else "No answer generated."

            # Parse nếu dùng structured output (nếu bạn có parser)
            if use_structured_output:
                try:
                    parsed = self.response_parser.parse_structured_response(raw_answer)
                except Exception:
                    parsed = {"answer": raw_answer}
            else:
                parsed = {"answer": raw_answer}

            # Thêm sources (giống Ollama)
            sources = []
            seen_urls = set()
            for chunk in chunks:
                arxiv_id = chunk.get("arxiv_id")
                if arxiv_id:
                    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
                    if pdf_url not in seen_urls:
                        sources.append(pdf_url)
                        seen_urls.add(pdf_url)

            citations = list({c.get("arxiv_id") for c in chunks if c.get("arxiv_id")})[:5]

            # Gộp kết quả
            result = {
                "answer": parsed.get("answer", raw_answer),
                "sources": sources,
                "citations": citations,
                "model": model_name,
            }
            if "confidence" in parsed:
                result["confidence"] = parsed["confidence"]

            return result

        except Exception as e:
            logger.error(f"Error generating RAG answer with Gemini: {e}")
            raise GeminiException(f"Failed to generate RAG answer: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming RAG answer - tương thích 100% với OllamaClient.generate_rag_answer_stream
        """
        try:
            model_name = model or self.default_model
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            logger.info(f"Starting streaming RAG answer with Gemini: {model_name}")

            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                generation_config=self.generation_config,
                stream=True,
            )

            for chunk in response:
                if chunk.text:
                    yield {"response": chunk.text}

        except Exception as e:
            logger.error(f"Error in Gemini streaming RAG: {e}")
            raise GeminiException(f"Streaming failed: {e}")
