from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    offset: int
    limit: int


class ErrorDetail(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    type: str = "error"
    error: ErrorDetail
