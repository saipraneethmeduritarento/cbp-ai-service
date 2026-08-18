from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import uuid

# Course recommendation schemas
class RecommendedCourseBase(BaseModel):
    """Base schema for Recommended Course"""
    role_mapping_id: uuid.UUID = Field(..., description="ID of the associated role mapping")
    # actual_courses: List[Dict[str, Any]] = Field(default=[], description="All courses found in search")
    filtered_courses: List[Dict[str, Any]] = Field(default=[], description="Filtered course recommendations")

class RecommendedCourseResponse(RecommendedCourseBase):
    """Schema for Recommended Course response"""
    id: uuid.UUID = Field(..., description="Unique identifier")
    user_id: uuid.UUID = Field(..., description="User ID")
    status: str = Field(..., description="Status")
    is_existing: bool = Field(default=False, description="Indicates whether the recommendation already existed (true) or was newly generated (false).")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }
class RecommendCourseCreate(BaseModel):
    """ Course Recommendation Generate"""
    role_mapping_id: uuid.UUID = Field(..., description="ID of the associated role mapping")


class BulkRecommendCourseCreate(BaseModel):
    """Bulk course recommendation generation request, scoped by state/department.

    Role mappings are derived server-side (all COMPLETED role mappings owned by the
    current user under this state/department), not supplied as an explicit id list.
    """
    state_center_id: str = Field(..., min_length=1, description="State/Center ID to generate recommendations for")
    department_id: Optional[str] = Field(
        default=None,
        description="Department ID to scope to. Omit to target org-level role mappings (department_id IS NULL)."
    )


class BulkRecommendationItemStatus(str, Enum):
    QUEUED = "QUEUED"
    ALREADY_IN_PROGRESS = "ALREADY_IN_PROGRESS"
    ALREADY_COMPLETED = "ALREADY_COMPLETED"
    ALREADY_FAILED = "ALREADY_FAILED"


class BulkRecommendationItemResult(BaseModel):
    role_mapping_id: uuid.UUID
    designation_name: Optional[str] = None
    status: BulkRecommendationItemStatus
    recommendation_id: Optional[uuid.UUID] = None

    class Config:
        json_encoders = {uuid.UUID: lambda v: str(v)}


class BulkGenerateCourseRecommendationsResponse(BaseModel):
    state_center_id: str
    department_id: Optional[str] = None
    total_role_mappings: int
    queued_count: int
    already_existing_count: int
    items: List[BulkRecommendationItemResult]

