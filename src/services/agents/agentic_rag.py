import logging
import time
import json
import re
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.runtime import Runtime
from langchain_mcp_adapters.tools import load_mcp_tools

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.nvidia.client import NvidiaClient
from src.services.neo4j import queries as neo4j_queries
from src.services.neo4j.client import Neo4jClient
from src.services.opensearch.client import OpenSearchClient

from .config import GraphConfig
from .context import Context
from .nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
    continue_after_guardrail,
    ainvoke_should_retrieve_step,
    route_after_should_retrieve,
    ainvoke_rerank_documents_step
)
from .state import AgentState

logger = logging.getLogger(__name__)


def extract_quoted_titles(query: str) -> List[str]:
    """Extract paper titles wrapped in single or double quotes."""
    if not query:
        return []
    matches = re.findall(r"['\"]([^'\"]+)['\"]", query)
    cleaned: List[str] = []
    for match in matches:
        title = match.strip()
        if title and title not in cleaned:
            cleaned.append(title)
    return cleaned


def is_shared_citation_query(query: str) -> bool:
    """Detect queries asking for references cited by both of two papers."""
    lowered = (query or "").lower()
    titles = extract_quoted_titles(query)
    citation_markers = ("cited by both", "cite", "cites", "cited", "bibliography", "references", "shared citations")
    return len(titles) >= 2 and "both" in lowered and any(marker in lowered for marker in citation_markers)


class AgenticRAGService:
    """Agentic RAG service 

    This implementation uses:
    - context_schema for dependency injection
    - Runtime[Context] for type-safe access in nodes
    - Direct client invocation (no pre-built runnables)
    - Lightweight nodes as pure functions
    """

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        neo4j_client: Optional[Neo4jClient],
        nvidia_client: NvidiaClient,
        embeddings_client: JinaEmbeddingsClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        graph_config: Optional[GraphConfig] = None,
        mcp_client: Optional[MultiServerMCPClient] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """Initialize agentic RAG service.

        :param opensearch_client: Client for document search
        :param ollama_client: Client for LLM generation
        :param embeddings_client: Client for embeddings
        :param langfuse_tracer: Optional Langfuse tracer
        :param graph_config: Configuration for graph execution
        :param checkpointer: LangGraph checkpointer (e.g. AsyncPostgresSaver). Defaults to InMemorySaver.
        """
        self.opensearch = opensearch_client
        self.neo4j = neo4j_client
        self.nvidia = nvidia_client

        self.embeddings = embeddings_client
        self.langfuse_tracer = langfuse_tracer
        self.graph_config = graph_config or GraphConfig()
        self._checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        # MCP client for tool execution
        if mcp_client is None:
            self.mcp_client = MultiServerMCPClient({
                "arxiv-tools": {
                    "transport": "http",           
                    "url": "http://fast-mcp-server:8100/mcp",
                },
            })
        else:
            self.mcp_client = mcp_client

        logger.info("Initializing AgenticRAGService with configuration:")
        logger.info(f"  Model: {self.graph_config.model}")
        logger.info(f"  Top-k: {self.graph_config.top_k}")
        logger.info(f"  Hybrid search: {self.graph_config.use_hybrid}")
        logger.info(f"  Neo4j enrichment: {self.neo4j is not None}")
        logger.info(f"  Max retrieval attempts: {self.graph_config.max_retrieval_attempts}")
        logger.info(f"  Guardrail threshold: {self.graph_config.guardrail_threshold}")

        self.graph = self._build_graph()
        logger.info("✓ AgenticRAGService initialized successfully")

    def _build_graph(self):
        """Build and compile the LangGraph workflow.

        Uses context_schema for type-safe dependency injection.
        Nodes are lightweight functions that receive Runtime[Context].

        :returns: Compiled graph ready for invocation
        """
        logger.info("Building LangGraph workflow with context_schema")

        # Create workflow with AgentState and Context schema
        workflow = StateGraph(AgentState, context_schema=Context)

        def _format_graph_facts(relations: List[dict]) -> str:
            if not relations:
                return ""

            lines = ["Graph facts:"]
            for rel in relations[:10]:
                rel_type = rel.get("rel_type", "RELATED_TO")
                obj_labels = ", ".join(rel.get("obj_labels") or [])
                obj_props = rel.get("obj_props") or {}
                obj_preview = ", ".join(
                    f"{key}={value}"
                    for key, value in list(obj_props.items())[:3]
                )
                lines.append(f"- {rel_type} -> [{obj_labels}] {obj_preview}".strip())
            return "\n".join(lines)

        def _build_pdf_url(arxiv_id: Optional[str]) -> str:
            if not arxiv_id:
                return ""
            clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            return f"https://arxiv.org/pdf/{clean_id}.pdf"

        def _lookup_paper_by_title(
            title: str,
            opensearch_client: OpenSearchClient,
        ) -> Optional[dict]:
            try:
                results = opensearch_client.search_papers(query=title, size=5, latest=False)
            except Exception as e:
                logger.warning("Failed to resolve paper title '%s': %s", title, e)
                return None

            hits = results.get("hits", [])
            if not hits:
                return None

            def normalize(value: Optional[str]) -> str:
                return re.sub(r"\s+", " ", (value or "")).strip().lower()

            normalized_title = normalize(title)
            exact_match = next((hit for hit in hits if normalize(hit.get("title")) == normalized_title), None)
            selected = exact_match or hits[0]
            arxiv_id = selected.get("arxiv_id", "")

            return {
                "arxiv_id": arxiv_id,
                "title": selected.get("title", title),
                "authors": selected.get("authors", []),
                "url": selected.get("pdf_url") or _build_pdf_url(arxiv_id),
                "relevance_score": selected.get("score", 1.0),
            }

        def _resolve_shared_citation_query(
            query: str,
            opensearch_client: OpenSearchClient,
            neo4j_client: Optional[Neo4jClient],
        ) -> Optional[dict]:
            if not is_shared_citation_query(query) or not neo4j_client:
                return None

            titles = extract_quoted_titles(query)[:2]
            if len(titles) < 2:
                return None

            paper_a = _lookup_paper_by_title(titles[0], opensearch_client)
            paper_b = _lookup_paper_by_title(titles[1], opensearch_client)
            if not paper_a or not paper_b:
                logger.info("Shared-citation query detected but failed to resolve both titles: %s", titles)
                return None

            if not paper_a.get("arxiv_id") or not paper_b.get("arxiv_id"):
                logger.info("Shared-citation query detected but one resolved paper lacks arxiv_id")
                return None

            cypher = (
                "MATCH (a {arxiv_id:$a}), (b {arxiv_id:$b})\n"
                "MATCH (a)-[:CITES|CITES_PAPER]->(r)<-[:CITES|CITES_PAPER]-(b)\n"
                "RETURN DISTINCT r.title AS title, r.arxiv_id AS arxiv_id, labels(r) AS labels LIMIT 200"
            )
            rows = neo4j_client.execute_read(
                cypher,
                {"a": paper_a["arxiv_id"], "b": paper_b["arxiv_id"]},
            )

            shared_docs = []
            for row in rows:
                cited_arxiv_id = row.get("arxiv_id") or ""
                cited_title = row.get("title") or "Untitled reference"
                shared_docs.append(
                    {
                        "page_content": (
                            f"Shared citation cited by both '{paper_a['title']}' and '{paper_b['title']}': "
                            f"{cited_title}"
                            + (f" (arXiv:{cited_arxiv_id})" if cited_arxiv_id else "")
                        ),
                        "metadata": {
                            "arxiv_id": cited_arxiv_id,
                            "title": cited_title,
                            "authors": [],
                            "score": 1.0,
                            "source": _build_pdf_url(cited_arxiv_id),
                            "section": "shared_citation",
                            "search_mode": "graph",
                            "top_k": len(rows),
                            "source_papers": [paper_a["arxiv_id"], paper_b["arxiv_id"]],
                            "labels": row.get("labels", []),
                        },
                    }
                )

            if not shared_docs:
                shared_docs.append(
                    {
                        "page_content": (
                            f"No shared citations found between '{paper_a['title']}' and '{paper_b['title']}'."
                        ),
                        "metadata": {
                            "arxiv_id": "",
                            "title": "No shared citations found",
                            "authors": [],
                            "score": 0.0,
                            "source": "",
                            "section": "shared_citation",
                            "search_mode": "graph",
                            "top_k": 0,
                            "source_papers": [paper_a["arxiv_id"], paper_b["arxiv_id"]],
                            "labels": [],
                        },
                    }
                )

            return {
                "relevant_sources": [paper_a, paper_b],
                "retrieved_docs": shared_docs,
                "neo4j_attempted": True,
                "graph_enriched_arxiv_ids": [paper_a["arxiv_id"], paper_b["arxiv_id"]],
            }

        def _enrich_with_neo4j(
            docs: List[dict],
            neo4j_client: Optional[Neo4jClient],
        ) -> tuple[List[dict], List[str], bool]:
            if not neo4j_client or not docs:
                logger.info(
                    "Skipping Neo4j enrichment: neo4j_client=%s docs=%s",
                    bool(neo4j_client),
                    len(docs),
                )
                return docs, [], False

            arxiv_ids = [
                doc.get("metadata", {}).get("arxiv_id")
                for doc in docs
                if doc.get("metadata", {}).get("arxiv_id")
            ]
            if not arxiv_ids:
                logger.info("Skipping Neo4j enrichment: no arxiv_ids found in retrieved docs")
                return docs, [], False

            enriched_ids: List[str] = []
            neo4j_attempted = True

            try:
                logger.info(
                    "Querying Neo4j for graph facts: arxiv_ids=%s",
                    arxiv_ids,
                )
                query = neo4j_queries.build_papers_relations_query()
                rows = neo4j_client.execute_read(query, {"ids": arxiv_ids})
                logger.info(
                    "Neo4j returned %s row(s) for arxiv_ids=%s",
                    len(rows),
                    arxiv_ids,
                )
                facts_map = {}
                for row in rows:
                    simplified = neo4j_queries.simplify_relations_row(row)
                    facts_map[simplified.get("arxiv_id")] = simplified.get("relations", [])

                for doc in docs:
                    metadata = doc.setdefault("metadata", {})
                    arxiv_id = metadata.get("arxiv_id")
                    relations = facts_map.get(arxiv_id, [])
                    if not relations:
                        continue

                    metadata["graph_facts"] = relations
                    enriched_ids.append(arxiv_id)
                    graph_facts_text = _format_graph_facts(relations)

                    page_content = (
                        doc.get("page_content")
                        or doc.get("text")
                        or doc.get("chunk_text")
                        or doc.get("content")
                        or ""
                    )
                    if graph_facts_text and graph_facts_text not in page_content:
                        doc["page_content"] = f"{page_content}\n\n{graph_facts_text}".strip()

                logger.info(
                    "Enriched %s retrieved documents with Neo4j graph facts",
                    len(enriched_ids),
                )
            except Exception as e:
                logger.warning("Neo4j enrichment failed; continuing without graph facts: %s", e)

            return docs, enriched_ids, neo4j_attempted

        # Create tools (these still need to be created upfront for ToolNode)
        async def tool_node(state: dict, runtime: Runtime[Context]):
            tool_calls = state["messages"][-1].tool_calls
            result = []
            relevant_sources = []
            observations = []
            enriched_ids: List[str] = []
            neo4j_attempted = False
            for call in tool_calls:
                tool = runtime.context.tools_by_name[call["name"]]
                if tool.name=="retrieve_papers":
                    query = call["args"].get("query", "")
                    shared_citation_result = _resolve_shared_citation_query(
                        query,
                        runtime.context.opensearch_client,
                        runtime.context.neo4j_client,
                    )
                    if shared_citation_result is not None:
                        observations = shared_citation_result["retrieved_docs"]
                        relevant_sources = shared_citation_result["relevant_sources"]
                        neo4j_attempted = shared_citation_result.get("neo4j_attempted", False)
                        enriched_ids = shared_citation_result.get("graph_enriched_arxiv_ids", [])
                    else:
                        observations = await tool.ainvoke(call["args"])
                        observations = observations[0]['text']
                        observations = json.loads(observations)
                        observations, enriched_ids, neo4j_attempted = _enrich_with_neo4j(
                            observations,
                            runtime.context.neo4j_client,
                        )

                        relevant_sources = [
                            {
                                "arxiv_id": obs['metadata'].get("arxiv_id"),
                                "title": obs['metadata'].get("title"),
                                "authors": obs['metadata'].get("authors"),
                                "url": obs['metadata'].get("source"),
                                "relevance_score": obs['metadata'].get("top_k"),
                            }
                            for obs in observations
                        ]
                
                # result.append(ToolMessage(content=observations, tool_call_id=tool_call["id"]))
            return {
                "messages": result,
                "relevant_sources": relevant_sources,
                "retrieved_docs": observations,
                "metadata": {
                    "neo4j_attempted": neo4j_attempted,
                    "used_neo4j": bool(enriched_ids),
                    "graph_enriched_docs": len(enriched_ids),
                    "graph_enriched_arxiv_ids": enriched_ids,
                },
            }

        # Add nodes (just function references - no closures needed!)
        logger.info("Adding nodes to workflow graph")
        workflow.add_node("guardrail", ainvoke_guardrail_step)
        workflow.add_node("out_of_scope", ainvoke_out_of_scope_step)
        workflow.add_node("retrieve", ainvoke_retrieve_step)
        workflow.add_node("tool_retrieve", tool_node)
        # workflow.add_node("tool_retrieve", ToolNode(tools))
        workflow.add_node("grade_documents", ainvoke_grade_documents_step)
        workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
        workflow.add_node("generate_answer", ainvoke_generate_answer_step)
        workflow.add_node("should_retrieve", ainvoke_should_retrieve_step)
        workflow.add_node("rerank", ainvoke_rerank_documents_step)
        
        # Add edges
        logger.info("Configuring graph edges and routing logic")

        # Start → guardrail validation
        workflow.add_edge(START, "guardrail")

        # Guardrail → route based on score
        workflow.add_conditional_edges(
            "guardrail",
            continue_after_guardrail,
            {
                "continue": "should_retrieve",
                "out_of_scope": "out_of_scope",
            },
        )
        workflow.add_conditional_edges(
            "should_retrieve",
            route_after_should_retrieve,
            {
                "generate_answer": "generate_answer",
                "retrieve": "retrieve",
            },
        )
        # Out of scope → END
        workflow.add_edge("out_of_scope", END)

        # Retrieve node creates tool call
        workflow.add_conditional_edges(
            "retrieve",
            tools_condition,  # Nếu có tool_calls → "tools", không có → END
            {
                "tools": "tool_retrieve",
                END: "generate_answer",  # QUAN TRỌNG: Khi skip retrieve → đi thẳng generate_answer
            },
        )

        # After tool retrieval → grade documents
        workflow.add_edge("tool_retrieve", "rerank")
        workflow.add_edge("rerank", "grade_documents")

        # After grading → route based on relevance
        workflow.add_conditional_edges(
            "grade_documents",
            lambda state: state.get("routing_decision", "generate_answer"),
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
            },
        )

        # After rewriting → try retrieve again
        workflow.add_edge("rewrite_query", "retrieve")

        # After answer generation → done
        workflow.add_edge("generate_answer", END)

        # Compile graph
        logger.info("Compiling LangGraph workflow")
        compiled_graph = workflow.compile(checkpointer=self._checkpointer)
        logger.info("✓ Graph compilation successful")

        return compiled_graph

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model: Optional[str] = None,
        thread_id: Optional[str] = "1",
        prompt_on_empty: bool = False,
    ) -> dict:
        """Ask a question using agentic RAG.

        :param query: User question
        :param user_id: User identifier for tracing
        :param model: Optional model override
        :returns: Dictionary with answer, sources, reasoning steps, and metadata
        :raises ValueError: If query is empty
        """
        model_to_use = model or self.graph_config.model

        logger.info("=" * 80)
        logger.info("Starting Agentic RAG Request")
        logger.info(f"Query: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Model: {model_to_use}")
        logger.info("=" * 80)

        # Validate input
        if not query or len(query.strip()) == 0:
            logger.error("Empty query received")
            raise ValueError("Query cannot be empty")

        # Create trace if Langfuse is enabled (v3 SDK)
        trace = None
        if self.langfuse_tracer and self.langfuse_tracer.client:
            logger.info("Creating Langfuse trace (v3 SDK)")
            metadata = {
                "env": self.graph_config.settings.environment,
                "service": "agentic_rag",
                "top_k": self.graph_config.top_k,
                "use_hybrid": self.graph_config.use_hybrid,
                "model": model_to_use,
            }
            # V3 SDK: start_as_current_observation (start_as_current_span removed in newer SDKs)
            trace = self.langfuse_tracer.client.start_as_current_observation(
                name="agentic_rag_request",
                as_type="span",
            )

        # Use proper context manager pattern
        async def _execute_with_trace():
            """Execute the workflow with or without tracing context."""
            if trace is not None:
                with trace as trace_obj:
                    trace_obj.update(
                        input={"query": query},
                        metadata=metadata,
                        user_id=user_id,
                        session_id=f"session_{user_id}",
                    )
                    logger.debug(f"Trace created: {trace_obj}")
                    return await self._run_workflow(query, model_to_use, user_id, trace_obj, thread_id)
            else:
                return await self._run_workflow(query, model_to_use, user_id, None, thread_id)

        try:
            async with self.mcp_client.session("arxiv-tools") as session:
                tools = await load_mcp_tools(session)
                self.tools_by_name = {tool.name: tool for tool in tools}

                result = await _execute_with_trace()

                # If no retrieved context/sources and interactive prompt requested,
                # ask the human whether to run web_search tool.
                if prompt_on_empty:
                    sources = result.get("sources", [])
                    retrieved = result.get("retrieved_contexts", [])
                    if not sources and (not retrieved or all(len(r.strip()) == 0 for r in retrieved)):
                        # prompt user in thread-safe way
                        loop = asyncio.get_running_loop()
                        try:
                            answer = await loop.run_in_executor(None, input, "No local data found. Run web search? (y/N): ")
                        except Exception:
                            answer = "n"
                        if answer and answer.strip().lower().startswith("y"):
                            web_tool = self.tools_by_name.get("web_search")
                            if web_tool:
                                try:
                                    web_res = await web_tool.ainvoke({"query": query})
                                    # web_res may be a dict/str depending on tool; normalize
                                    summary = web_res if isinstance(web_res, str) else str(web_res)
                                    # append web search summary to answer payload
                                    result.setdefault("web_search_summary", summary)
                                except Exception:
                                    logger.exception("Failed to invoke web_search tool")

                return result

        except Exception as e:
            logger.error(f"Error in Agentic RAG execution: {str(e)}")
            logger.exception("Full traceback:")
            raise

    async def _run_workflow(self, query: str, model_to_use: str, user_id: str, trace, thread_id) -> dict:
        """Execute the workflow with the given trace context."""
        try:
            start_time = time.time()

            logger.info("Invoking LangGraph workflow")

            # State initialization
            state_input = {
                "messages": [HumanMessage(content=query)],
                "retrieval_attempts": 0,
                "guardrail_result": None,
                "routing_decision": None,
                "sources": None,
                "relevant_sources": [],
                "relevant_tool_artefacts": None,
                "grading_results": [],
                "metadata": {},
                "original_query": None,
                "rewritten_query": None,
            }

            # Runtime context (dependencies)
            runtime_context = Context(
                # ollama_client=self.ollama,
                nvidia_client=self.nvidia,
                opensearch_client=self.opensearch,
                neo4j_client=self.neo4j,
                embeddings_client=self.embeddings,
                langfuse_tracer=self.langfuse_tracer,
                trace=trace,
                langfuse_enabled=self.langfuse_tracer is not None and self.langfuse_tracer.client is not None,
                model_name=model_to_use,
                temperature=self.graph_config.temperature,
                top_k=self.graph_config.top_k,
                max_retrieval_attempts=self.graph_config.max_retrieval_attempts,
                guardrail_threshold=self.graph_config.guardrail_threshold,
                tools_by_name=self.tools_by_name,
            )
            
            config = {
                "configurable": {"thread_id": thread_id},
            }


            # Add CallbackHandler for automatic LLM tracing
            # IMPORTANT: CallbackHandler automatically inherits the current span context
            # Since we're inside start_as_current_observation, it will be linked automatically
            if self.langfuse_tracer and trace:
                try:
                    # V3 SDK: CallbackHandler() automatically uses current trace context
                    # No need to pass trace explicitly - it's handled by context propagation
                    callback_handler = CallbackHandler()
                    # config["callbacks"] = [callback_handler]
                    config = {
                        "configurable": {"thread_id": thread_id},
                        "callbacks": [callback_handler]
                    }
                    logger.info("✓ CallbackHandler added (will auto-link to current trace)")
                except Exception as e:
                    logger.warning(f"Failed to create CallbackHandler: {e}")

            result = await self.graph.ainvoke(
                state_input,
                config=config,
                context=runtime_context,
            )
            
            trace_id = self.langfuse_tracer.get_trace_id()
            logger.warning(f"Trace id: {trace_id}")

            execution_time = time.time() - start_time
            logger.info(f"✓ Graph execution completed in {execution_time:.2f}s")

            # Extract results
            answer = self._extract_answer(result)
            sources = self._extract_sources(result)
            retrieval_attempts = result.get("retrieval_attempts", 0)
            reasoning_steps = self._extract_reasoning_steps(result)

            # Update trace (cleanup handled by context manager)
            if trace:
                trace.update(
                    output={
                        "answer": answer,
                        "sources_count": len(sources),
                        "retrieval_attempts": retrieval_attempts,
                        "reasoning_steps": reasoning_steps,
                        "execution_time": execution_time,
                    }
                )
                self.langfuse_tracer.flush()

            logger.info("=" * 80)
            logger.info("Agentic RAG Request Completed Successfully")
            logger.info(f"Answer length: {len(answer)} characters")
            logger.info(f"Sources found: {len(sources)}")
            logger.info(f"Retrieval attempts: {retrieval_attempts}")
            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info("=" * 80)

            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "retrieved_contexts": [
                    (
                        doc.get("page_content")
                        or doc.get("text")
                        or doc.get("chunk_text")
                        or doc.get("content")
                        or ""
                    )
                    if isinstance(doc, dict)
                    else str(doc)
                    for doc in result.get("retrieved_docs", [])
                ],
                "reasoning_steps": reasoning_steps,
                "retrieval_attempts": retrieval_attempts,
                "rewritten_query": result.get("rewritten_query"),
                "execution_time": execution_time,
                "guardrail_score": result.get("guardrail_result").score if result.get("guardrail_result") else None,
                "trace_id": trace_id,
                "neo4j_attempted": result.get("metadata", {}).get("neo4j_attempted", False),
                "used_neo4j": result.get("metadata", {}).get("used_neo4j", False),
                "graph_enriched_docs": result.get("metadata", {}).get("graph_enriched_docs", 0),
                "graph_enriched_arxiv_ids": result.get("metadata", {}).get("graph_enriched_arxiv_ids", []),
                # If no local sources were found, tell the caller we can run a web search
                # The frontend can surface `web_search_prompt` to the user and call
                # the suggested `web_search_action` if they agree.
                "needs_web_search": (len(sources) == 0 and all(len(c.strip()) == 0 for c in [
                    (
                        doc if isinstance(doc, str) else str(doc)
                    ) for doc in result.get("retrieved_docs", [])
                ])),
                "web_search_prompt": (
                    "I’m sorry, but the set of retrieved documents does not contain any information about the authors of the article “Very deep convolutional networks for large‑scale image recognition.” "
                    "Consequently, I cannot provide an answer based on the available sources. If you can supply a relevant paper or additional context, I’ll be happy to help further. Do you want me to search the web?"
                ),
                "web_search_action": {"tool": "web_search", "args": {"query": query}},
            }

        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            logger.exception("Full traceback:")

            # Update trace with error (cleanup handled by context manager)
            if trace:
                trace.update(output={"error": str(e)}, level="ERROR")
                self.langfuse_tracer.flush()

            raise

    def _extract_answer(self, result: dict) -> str:
        """Extract final answer from graph result."""
        messages = result.get("messages", [])
        if not messages:
            return "No answer generated."

        final_message = messages[-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)

    def _extract_sources(self, result: dict) -> List[dict]:
        """Extract sources from graph result."""
        sources = []
        seen = set()
        relevant_sources = result.get("relevant_sources", [])

        for source in relevant_sources:
            if hasattr(source, "to_dict"):
                source = source.to_dict()
            elif isinstance(source, dict):
                source = source
            else:
                continue

            url = source.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append(url)

        return sources

    def _extract_reasoning_steps(self, result: dict) -> List[str]:
        """Extract reasoning steps from graph result."""
        steps = []
        retrieval_attempts = result.get("retrieval_attempts", 0)
        guardrail_result = result.get("guardrail_result")
        grading_results = result.get("grading_results", [])

        if guardrail_result:
            steps.append(f"Validated query scope (score: {guardrail_result.score}/100)")

        if retrieval_attempts > 0:
            steps.append(f"Retrieved documents ({retrieval_attempts} attempt(s))")

        if grading_results:
            relevant_count = sum(1 for g in grading_results if g.is_relevant)
            steps.append(f"Graded documents ({relevant_count} relevant)")

        if result.get("rewritten_query"):
            steps.append("Rewritten query for better results")

        steps.append("Generated answer from context")

        return steps

    def get_graph_visualization(self) -> bytes:
        """Get the LangGraph workflow visualization as PNG.

        This method generates a visual representation of the graph workflow
        using mermaid diagram format, then converts it to PNG.

        :returns: PNG image bytes
        :raises ImportError: If required dependencies (pygraphviz/graphviz) are not installed
        :raises Exception: If graph visualization generation fails

        Example:
            >>> service = AgenticRAGService(...)
            >>> png_bytes = service.get_graph_visualization()
            >>> with open("graph.png", "wb") as f:
            ...     f.write(png_bytes)
        """
        try:
            logger.info("Generating graph visualization as PNG")
            png_bytes = self.graph.get_graph().draw_mermaid_png()
            logger.info(f"✓ Generated PNG visualization ({len(png_bytes)} bytes)")
            return png_bytes
        except ImportError as e:
            logger.error(f"Failed to generate visualization - missing dependencies: {e}")
            logger.error("Install with: pip install pygraphviz or apt-get install graphviz")
            raise ImportError(
                "Graph visualization requires pygraphviz. "
                "Install with: pip install pygraphviz (requires graphviz system package)"
            ) from e
        except Exception as e:
            logger.error(f"Failed to generate graph visualization: {e}")
            raise

    def get_graph_mermaid(self) -> str:
        """Get the LangGraph workflow as a mermaid diagram string.

        This method generates the graph workflow representation in mermaid
        diagram syntax, which can be rendered in markdown or mermaid viewers.

        :returns: Mermaid diagram syntax as string

        Example:
            >>> service = AgenticRAGService(...)
            >>> mermaid = service.get_graph_mermaid()
            >>> print(mermaid)
            graph TD
                __start__ --> guardrail
                ...
        """
        try:
            logger.info("Generating graph as mermaid diagram")
            mermaid_str = self.graph.get_graph().draw_mermaid()
            logger.info(f"✓ Generated mermaid diagram ({len(mermaid_str)} characters)")
            return mermaid_str
        except Exception as e:
            logger.error(f"Failed to generate mermaid diagram: {e}")
            raise

    def get_graph_ascii(self) -> str:
        """Get ASCII representation of the graph.

        This method generates a simple ASCII art representation of the
        graph structure, useful for quick inspection in terminals.

        :returns: ASCII art representation of the graph

        Example:
            >>> service = AgenticRAGService(...)
            >>> print(service.get_graph_ascii())
        """
        try:
            logger.info("Generating ASCII graph representation")
            ascii_str = self.graph.get_graph().print_ascii()
            logger.info("✓ Generated ASCII graph representation")
            return ascii_str
        except Exception as e:
            logger.error(f"Failed to generate ASCII graph: {e}")
            raise
