from src.config import Settings, get_settings
from src.services.evaluation.service import EvaluationService
from src.services.langfuse.client import LangfuseTracer
from src.services.nvidia.client import NvidiaClient


def make_evaluation_service(
    settings: Settings | None = None,
    nvidia_client: NvidiaClient | None = None,
    langfuse_tracer: LangfuseTracer | None = None,
) -> EvaluationService:
    app_settings = settings or get_settings()
    return EvaluationService(
        settings=app_settings,
        nvidia_client=nvidia_client,
        langfuse_tracer=langfuse_tracer,
    )
