import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.config import get_settings
# from src.dependencies import SessionDep
from src.repositories.paper import PaperRepository
from src.schemas.api.upload import PaperUploadResponse
from src.schemas.arxiv.paper import PaperCreate
from src.api.deps import CurrentUser,SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/papers", tags=["papers"])


@router.post("/upload", response_model=PaperUploadResponse)
async def upload_paper(
    db_session: SessionDep,
    current_user: CurrentUser,
    session_id: str = Form(..., description="Session ID for scoping user-uploaded papers"),  # <-- add this
    file: UploadFile = File(..., description="PDF file to upload"),
):
    """
    Upload a PDF paper. Airflow will parse it and extract metadata automatically.
    
    The uploaded paper will be:
    1. Saved to shared volume
    2. Airflow DAG triggered to parse PDF and extract metadata
    3. Paper stored in DB with metadata extracted from PDF
    4. Indexed in OpenSearch for search
    
    Args:
        file: PDF file to upload (filename without .pdf will be used as arxiv_id)
    
    Returns:
        PaperUploadResponse with upload status
    """
    settings = get_settings()
    
    # Validate file type
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Use filename (without .pdf) as arxiv_id
    arxiv_id = file.filename.replace(".pdf", "").replace(".PDF", "").strip()
    if not arxiv_id:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    logger.info(f"Uploading PDF with arxiv_id (from filename): {arxiv_id}")
    
    # Ensure upload directory exists
    upload_dir = Path("/app/data/user_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = upload_dir / f"{arxiv_id}.pdf"
    
    try:
        # Save uploaded file to shared volume
        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved uploaded PDF: {pdf_path} ({len(content)} bytes)")
        
        # Trigger Airflow DAG for async processing (parse + extract metadata + store + index)
        airflow_url = settings.airflow.webserver_url
        dag_id = "user_paper_processing"
        
        dag_triggered = False
        dag_run_id = None
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns",
                    json={
                        "conf": {
                            "arxiv_id": arxiv_id,
                            "pdf_path": str(pdf_path),
                            "session_id": session_id,
                        }
                    },
                    auth=(settings.airflow.username, settings.airflow.password),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                dag_run = response.json()
                dag_run_id = dag_run.get("dag_run_id")
                dag_triggered = True
                logger.info(f"Successfully triggered Airflow DAG run: {dag_run_id}")
        except httpx.ConnectError as e:
            logger.warning(f"Failed to connect to Airflow: {e}. PDF saved, can trigger manually.")
        except Exception as e:
            logger.warning(f"Failed to trigger Airflow DAG: {e}. PDF saved, can trigger manually.")
        
        if dag_triggered:
            message = f"PDF uploaded. Processing in Airflow (DAG run: {dag_run_id}). Metadata will be extracted from PDF."
        else:
            message = (
                f"PDF uploaded but Airflow trigger failed. PDF saved at {pdf_path}. "
                f"Trigger DAG '{dag_id}' manually with conf: "
                f"{{'arxiv_id': '{arxiv_id}', 'pdf_path': '{pdf_path}', 'session_id': '{session_id}'}}"
            )
        # === HARD-CODE PAPER_ACCESS FOR DEBUG ===
        try:
            from src.repositories.paper_access import PaperAccessRepository

            access_repo = PaperAccessRepository(db_session)

            fake_access = access_repo.grant_session_access(arxiv_id, session_id, session_id)

            logger.warning(
                "[DEBUG] Forced PaperAccess created on upload | "
                f"paper_id={fake_access.paper_id}, session_id={fake_access.subject_id}"
            )
        except Exception as e:
            logger.error(
                "[DEBUG] Failed to force PaperAccess on upload",
                exc_info=True,
            )
        # === END DEBUG ===

        return PaperUploadResponse(
            paper_id=None,  # Will be created by Airflow
            arxiv_id=arxiv_id,
            title="Processing...",  # Will be extracted by Airflow
            authors=["Processing..."],
            abstract="Processing...",
            chunks_indexed=0,
            message=message,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        if pdf_path.exists():
            pdf_path.unlink()
        logger.error(f"Error uploading paper: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload paper: {str(e)}")
