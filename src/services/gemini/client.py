import json
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

from src.config import Settings
from src.exceptions import GeminiConnectionError, GeminiException, GeminiTimeoutError
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, settings: Settings):
        api_key = settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        self.client = genai.Client(api_key=api_key)
        self.timeout = settings.gemini_timeout
        self.default_model = settings.gemini_model
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

        # TẠO CLIENT ASYNC CỐ ĐỊNH MỘT LẦN DUY NHẤT
        self.aio_client = self.client.aio  # Lưu lại để tái sử dụng nhiều lần

        self.generation_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=2048,
        )

        logger.info(f"GeminiClient initialized with model: {self.default_model}")

    async def health_check(self) -> Dict[str, Any]:
        """Check if Gemini service is healthy."""
        try:
            response = await self.client.models.generate_content(
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
        except Exception as e:
            raise GeminiConnectionError(f"Gemini health check failed: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Liệt kê các model Gemini có sẵn (dùng SDK chính thức).
        """
        try:
            models = await self.client.models.list()
            return [
                {
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description,
                    "supported_generation_methods": model.supported_generation_methods,
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Error listing Gemini models: {e}")
            raise GeminiException(f"Error listing models: {e}")

    async def generate(self, model: str = None, prompt: str = None, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate text - tương thích với OllamaClient.
        Hỗ trợ structured output JSON khi truyền kwargs['format'] = schema_dict.
        """
        try:
            model_name = model or self.default_model
            config = self.generation_config.copy()

            format_schema = kwargs.pop("format", None)
            structured = bool(format_schema)

            logger.info(
                f"Sending request to Gemini: model={model_name}, stream={stream}, "
                f"structured={structured}, extra_params={kwargs}"
            )

            # QUAN TRỌNG: Dùng self.aio_client cố định, KHÔNG dùng async with
            if stream:
                # Streaming: thu thập hết chunk rồi trả về full text (giữ tương thích cũ)
                full_text = ""
                response = await self.aio_client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                )
                async for chunk in response:
                    full_text += chunk.text or ""
                result = {"text": full_text}

            else:
                # Non-streaming
                response = await self.aio_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                text = response.text or ""

                if structured:
                    try:
                        parsed = json.loads(text)
                        result = {"parsed": parsed, "text": text}
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse structured JSON from Gemini")
                        result = {"text": text}
                else:
                    result = {"text": text}

            return result

        except Exception as e:
            logger.error(f"Error generating with Gemini: {e}")
            raise GeminiException(f"Error generating with Gemini: {e}")

    async def generate_stream(self, model: str = None, prompt: str = None, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
            try:
                model_name = model or self.default_model

                if kwargs:
                    logger.warning("Streaming hiện tại không hỗ trợ override config như temperature, top_p...")

                logger.info(f"Sending streaming request to Gemini: model={model_name}")

                # KHÔNG DÙNG async with nữa → dùng client cố định
                response = await self.aio_client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                )

                async for chunk in response:
                    yield {
                        "model": model_name,
                        "response": chunk.text or "",
                        "done": False,
                    }

                # Chunk cuối
                yield {
                    "model": model_name,
                    "response": "",
                    "done": True,
                }

            except Exception as e:
                logger.error(f"Error during streaming generation with Gemini: {e}")
                raise GeminiException(f"Error during streaming generation: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = None,
        use_structured_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate RAG answer - giữ nguyên logic cũ nhưng dùng Gemini SDK.
        """
        try:
            model_name = model or self.default_model

            if use_structured_output:
                prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)
                response = await self.generate(
                    model=model_name,
                    prompt=prompt_data["prompt"],
                    format=prompt_data["format"],  # truyền schema để ép JSON
                )
            else:
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)
                response = await self.generate(
                    model=model_name,
                    prompt=prompt,
                )

            # Lấy text hoặc parsed JSON
            if response.get("parsed"):
                parsed_response = response["parsed"]
            elif response.get("text"):
                raw_text = response["text"]
                parsed_response = self.response_parser.parse_structured_response(raw_text)
            else:
                raise GeminiException("No valid response from Gemini")

            # Bổ sung sources và citations nếu thiếu
            if not parsed_response.get("sources"):
                seen_urls = set()
                sources = []
                for chunk in chunks:
                    arxiv_id = chunk.get("arxiv_id")
                    if arxiv_id:
                        arxiv_id_clean = arxiv_id.split("v")[0]
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                        if pdf_url not in seen_urls:
                            sources.append(pdf_url)
                            seen_urls.add(pdf_url)
                parsed_response["sources"] = sources

            if not parsed_response.get("citations"):
                citations = list({chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")})
                parsed_response["citations"] = citations[:5]

            return parsed_response

        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise GeminiException(f"Failed to generate RAG answer: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming RAG answer - yield từng chunk.
        """
        try:
            model_name = model or self.default_model
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            async for chunk in self.generate_stream(model=model_name, prompt=prompt):
                yield chunk

        except Exception as e:
            logger.error(f"Error generating streaming RAG answer: {e}")
            raise GeminiException(f"Failed to generate streaming RAG answer: {e}")
