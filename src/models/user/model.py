# models.py
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, Float, DateTime, CHAR
from sqlalchemy.orm import relationship
from datetime import datetime

import uuid

from pydantic import EmailStr
from sqlmodel import Field, Relationship, SQLModel


# Shared properties
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)
    

