from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class PaperUploadRequest(BaseModel):
    """Optional metadata for uploaded papers."""
    
    title: Optional[str] = Field(None, description="Paper title (will be extracted from PDF if not provided)")
    authors: Optional[List[str]] = Field(None, description="List of authors (will be extracted from PDF if not provided)")
    abstract: Optional[str] = Field(None, description="Paper abstract (will be extracted from PDF if not provided)")
    categories: Optional[List[str]] = Field(default_factory=lambda: ["user-upload"], description="Paper categories")


class PaperUploadResponse(BaseModel):
    """Response after successful paper upload."""
    
    paper_id: Optional[UUID] = Field(None, description="ID of the uploaded paper (created by Airflow)")
    arxiv_id: str = Field(..., description="Unique identifier for the paper (user-upload-{uuid})")
    title: str = Field(..., description="Paper title (extracted from PDF by Airflow)")
    authors: List[str] = Field(..., description="List of authors (extracted from PDF by Airflow)")
    abstract: str = Field(..., description="Paper abstract (extracted from PDF by Airflow)")
    chunks_indexed: int = Field(..., description="Number of chunks indexed in OpenSearch")
    message: str = Field(..., description="Success message")
    
