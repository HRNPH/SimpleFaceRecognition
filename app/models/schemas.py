from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import Optional


class FaceRecognitionRequest(BaseModel):
    temperature: float = Field(
        ..., description="The temperature value of the individual."
    )
    rfid: str = Field(..., min_length=1, description="RFID identifier for the user.")
    image_base64: str = Field(
        ..., description="Base64 encoded image of the individual."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "temp": 36.5,
                "rfid": "1234567890",
                "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD... (truncated base64)",
            }
        }


class DefaultResponse(BaseModel):
    """Response model for the API, Also used for logging to firebase."""

    success: bool = Field(
        ..., description="Indicates whether the operation was successful."
    )
    message: Optional[str] = Field(
        None,
        description="Message for a successful operation, Reason for failure if the operation was not successful.",
    )
    similarity: Optional[float] = Field(
        None, description="Similarity score between the two faces."
    )
    threshold: Optional[float] = Field(
        None, description="Threshold value for the similarity score."
    )
    temperature: Optional[float] = Field(
        None, description="Temperature value of the individual."
    )
    rfid: Optional[str] = Field(None, description="RFID identifier for the user.")

    class Config:
        json_schema_extra = {
            "example": {"success": True, "message": "OK!", "reason": None},
            "examples": {
                "success": {
                    "summary": "Success Response",
                    "value": {"success": True, "message": "OK!", "reason": None},
                },
                "failure": {
                    "summary": "Failure Response",
                    "value": {
                        "success": False,
                        "message": None,
                        "reason": "Invalid RFID",
                    },
                },
            },
        }


# User
class UserCreationRequest(BaseModel):
    rfid: str = Field(..., min_length=1, description="RFID identifier for the user.")
    name: str = Field(..., min_length=1, description="Name of the user.")
    image_base64: str = Field(
        ..., description="Base64 encoded source face image of the individual."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "rfid": "1234567890",
                "name": "John Doe",
                "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD... (truncated base64)",
            }
        }


# Firebaselog
class FirebaseLog(DefaultResponse):  # Extend DefaultResponse
    createdAt: datetime = Field(
        ..., description="Time of creation of the log.", default_factory=datetime.now
    )
    image_base64: Optional[str] = Field(
        None, description="Base64 encoded image of the individual."
    )

    def to_firebase_dict(self):
        """Convert the object to a dictionary for Firebase."""
        return {
            **self.model_dump(
                exclude={"createdAt"}
            ),  # Include all fields except createdAt
            "createdAt": self.createdAt.isoformat(),  # Convert datetime to ISO 8601
        }
