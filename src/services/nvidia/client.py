import json
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

from src.config import Settings
from src.exceptions import NvidiaConnectionError, NvidiaException, NvidiaTimeoutError
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank
from langchain_core.messages import HumanMessage,AIMessage

logger = logging.getLogger(__name__)




class NvidiaClient:
    def __init__(self, settings: Settings):
        self.api_key = settings.nvidia_api_key
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is not configured")

        self.default_model = settings.nvidia_model
        self.timeout = settings.nvidia_timeout
        self.base_url = settings.nvidia_base_url
        
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()
        
        # Sử dụng nvidia_api_key để rõ ràng hơn (tương đương api_key)
        self.client = ChatNVIDIA(
            nvidia_api_key=self.api_key,
            model=self.default_model,
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            timeout=self.timeout,  # sẽ được truyền thành request_timeout
        )

        logger.info(f"NvidiaClient initialized with model: {self.default_model}")

    def get_langchain_model(self, model, temperature):
        client = ChatNVIDIA(
            nvidia_api_key=self.api_key,
            model=model,
            temperature=temperature,
            top_p=0.9,
            max_tokens=2048
        )
        return client
    def get_reranker(self, model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2", top_n: int = 8):
        """
        Trả về NVIDIARerank instance thay vì ChatNVIDIA
        """
        reranker = NVIDIARerank(
            nvidia_api_key=self.api_key,
            model=model,
            top_n=top_n,
            # Nếu dùng NIM local thì thêm: base_url=self.base_url
            # timeout=self.timeout nếu cần
        )
        logger.info(f"Reranker initialized with model: {model}, top_n: {top_n}")
        return reranker
    
    async def health_check(self):
            try:
                models = self.client.get_available_models()
                return {
                    "status": "healthy",
                    "model_count": len(models),
                }
            except Exception as e:
                raise NvidiaConnectionError(f"Nvidia health check failed: {e}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Liệt kê các model Nvidia có sẵn (sử dụng API key đã truyền trong __init__).
        """
        try:
            # Sử dụng chính instance để lấy list models → tự động dùng API key đã cấu hình
            models = self.client.get_available_models()
            
            return [
                {
                    "id": model.id,  # tên chính thức, ví dụ: "meta/llama3-70b-instruct"
                    "display_name": getattr(model, "display_name", model.id),
                    "description": getattr(model, "description", None),
                    "context_window": getattr(model, "max_tokens", None),  # thường là context length
                    "supports_tools": getattr(model, "supports_tools", False),
                    # Có thể thêm các field khác nếu cần
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Error listing Nvidia models: {e}")
            raise NvidiaException(f"Error listing models: {e}")

    async def generate(
        self,
        model: str = None,
        prompt: str = None,
        stream: bool = False,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate text - tương thích với OllamaClient.
        Hỗ trợ structured output JSON khi truyền kwargs['format'] = schema_dict.
        """
        try:
            model_name = model or self.default_model

            # Lấy schema nếu có yêu cầu structured output
            format_schema = kwargs.pop("format", None)
            structured = bool(format_schema)

            logger.info(
                f"Sending request to Nvidia: model={model_name}, stream={stream}, "
                f"structured={structured}, extra_params={kwargs}"
            )

            # Chuẩn bị messages (LangChain yêu cầu list messages)
            messages = [HumanMessage(content=prompt)]

            # Config chung cho invoke (có thể override bằng kwargs)
            invoke_kwargs = {
                "model": model_name,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "max_tokens": kwargs.get("max_tokens", 2048),
            }

            if structured and format_schema:
                # ChatNVIDIA hỗ trợ response_format JSON schema (từ phiên bản mới)
                invoke_kwargs["response_format"] = {
                    "type": "json_object",
                    "schema": format_schema
                }
            

            if stream:
                # Streaming: thu thập full text để tương thích cũ
                full_text = ""
                async for chunk in self.client.astream(messages, **invoke_kwargs):
                    chunk_text = chunk.content or ""
                    full_text += chunk_text

                result = {"text": full_text}

            else:
                # Non-streaming
                response: AIMessage = await self.client.ainvoke(messages, **invoke_kwargs)
                text = response.content.strip() if response.content else ""

                if structured:
                    try:
                        parsed = json.loads(text)
                        result = {"parsed": parsed, "text": text}
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse structured JSON from Nvidia response")
                        result = {"text": text}
                else:
                    result = {"text": text}

            return result

        except Exception as e:
            logger.error(f"Error generating with Nvidia: {e}")
            raise NvidiaException(f"Error generating with Nvidia: {e}")


    async def generate_stream(
        self,
        model: str = None,
        prompt: str = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming generation - yield từng chunk tương thích với giao diện cũ.
        """
        try:
            model_name = model or self.default_model

            if kwargs:
                logger.warning("Một số config override có thể không được áp dụng đầy đủ trong streaming")

            messages = [HumanMessage(content=prompt)]

            invoke_kwargs = {
                "model": model_name,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "max_tokens": kwargs.get("max_tokens", 2048),
            }

            logger.info(f"Sending streaming request to Nvidia: model={model_name}")

            async for chunk in self.client.astream(messages, **invoke_kwargs):
                yield {
                    "model": model_name,
                    "response": chunk.content or "",
                    "done": False,
                }

            # Chunk cuối để báo kết thúc
            yield {
                "model": model_name,
                "response": "",
                "done": True,
            }

        except Exception as e:
            logger.error(f"Error during streaming generation with Nvidia: {e}")
            raise NvidiaException(f"Error during streaming generation: {e}")


    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = None,
        use_structured_output: bool = True,
    ) -> Dict[str, Any]:
    #     """
    #     Generate RAG answer với structured output nếu cần.
    #     """
    #     try:
    #         model_name = model or self.default_model

    #         if use_structured_output:
    #             prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)
    #             response = await self.generate(
    #                 model=model_name,
    #                 prompt=prompt_data["prompt"],
    #                 format=prompt_data["format"],  # truyền schema JSON
    #             )
    #         else:
    #             prompt = self.prompt_builder.create_rag_prompt(query, chunks)
    #             response = await self.generate(
    #                 model=model_name,
    #                 prompt=prompt,
    #             )

    #         # Xử lý response
    #         if response.get("parsed"):
    #             parsed_response = response["parsed"]
    #         elif response.get("text"):
    #             raw_text = response["text"]
    #             parsed_response = self.response_parser.parse_structured_response(raw_text)
    #         else:
    #             raise NvidiaException("No valid response from Nvidia")

    #         # Bổ sung sources và citations nếu thiếu (giữ logic cũ)
    #         if not parsed_response.get("sources"):
    #             seen_urls = set()
    #             sources = []
    #             for chunk in chunks:
    #                 arxiv_id = chunk.get("arxiv_id")
    #                 if arxiv_id:
    #                     arxiv_id_clean = arxiv_id.split("v")[0]
    #                     pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
    #                     if pdf_url not in seen_urls:
    #                         sources.append(pdf_url)
    #                         seen_urls.add(pdf_url)
    #             parsed_response["sources"] = sources

    #         if not parsed_response.get("citations"):
    #             citations = list({chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")})
    #             parsed_response["citations"] = citations[:5]

    #         return parsed_response

    #     except Exception as e:
    #         logger.error(f"Error generating RAG answer: {e}")
    #         raise NvidiaException(f"Failed to generate RAG answer: {e}")

        """
        Generate a RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Model to use for generation
            use_structured_output: Whether to use Ollama's structured output feature

        Returns:
            Dictionary with answer, sources, confidence, and citations
        """
        try:
            if use_structured_output:
                # Use structured output with Pydantic model
                prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)

                # Generate with structured format
                response = await self.generate(
                    model=model,
                    prompt=prompt_data["prompt"],
                    temperature=0.7,
                    top_p=0.9,
                    format=prompt_data["format"],
                )
            else:
                # Fallback to plain text mode
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)

                # Generate without format restrictions
                response = await self.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.7,
                    top_p=0.9,
                )

            if response and "text" in response:
                answer_text = response["text"]
                logger.debug(f"Raw LLM response: {answer_text[:500]}")

                if use_structured_output:
                    # Try to parse structured response if enabled
                    parsed_response = self.response_parser.parse_structured_response(answer_text)
                    logger.debug(f"Parsed response: {parsed_response}")
                    return parsed_response
                else:
                    # For plain text response, build simple response structure
                    sources = []
                    logger.debug(f"Building sources from chunks for non-structured output{chunks}")
                    seen_urls = set()
                    for chunk in chunks:
                        arxiv_id = chunk.get("arxiv_id")
                        if arxiv_id:
                            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                            if pdf_url not in seen_urls:
                                sources.append(pdf_url)
                                seen_urls.add(pdf_url)

                    citations = list(set(chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")))

                    return {
                        "answer": answer_text,
                        "sources": sources,
                        "confidence": "medium",
                        "citations": citations[:5],
                    }
            else:
                raise NvidiaException("No response generated from Nvidia")

        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise NvidiaException(f"Failed to generate RAG answer: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming RAG answer - yield từng chunk text.
        """
        try:
            model_name = model or self.default_model
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            async for chunk in self.generate_stream(model=model_name, prompt=prompt):
                yield chunk

        except Exception as e:
            logger.error(f"Error generating streaming RAG answer: {e}")
            raise NvidiaException(f"Failed to generate streaming RAG answer: {e}")
