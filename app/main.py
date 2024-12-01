from dataclasses import dataclass
import datetime
from http.client import HTTPException
from typing import Annotated, Dict
from pydantic import BaseModel, UUID4
from litestar import Litestar, Request, Response, get, post
from litestar.openapi import OpenAPIConfig
from litestar.params import Body
from litestar import exceptions
from litestar.openapi.spec import Components, SecurityScheme, Tag
from litestar.openapi.plugins import SwaggerRenderPlugin
from dotenv import load_dotenv
from app.controller import FaceRecognitionController
from app.libs.firebase_logger import FirebaseAuthenticationLogs
from app.libs.auth import check_api_key
from app.libs.notification import LineNotify
import app.models.schemas as schemas
from app.libs.exception_handler import app_exception_handler
from app.db.prisma import db
from app.libs.s3 import S3Uploader
import os

load_dotenv()


@get("/", include_in_schema=True)
async def root() -> str:
    return "OK!"


controller = FaceRecognitionController()
firebase_logger = FirebaseAuthenticationLogs()
line_notify_client = LineNotify()
s3_client = S3Uploader()


@post(
    "/api/v1/inference",
    tags=["internal"],
    include_in_schema=True,
    status_code=200,
    security=[{"x-api-key": []}],
    content_media_type="application/json",
    content_encoding="utf-8",
    description="Perform face recognition on the input image",
)
async def inference(
    request: Request,
    data: schemas.FaceRecognitionRequest,
) -> schemas.DefaultResponse:
    check_api_key(request.headers)  # Raises HTTPException if API key is invalid
    response = await controller.process_face_recognition(data)

    # Logging and Notification
    # -- Construct log data
    time_stamp = datetime.datetime.now()
    image_uploaded = (
        s3_client.upload(
            image_base64=data.image_base64,
            file_path=f"images/{response.rfid}/{time_stamp.isoformat()}.png",
        )
        if data.image_base64
        else None
    )
    log_data = schemas.FirebaseLog(
        **response.model_dump(),
        image_base64=data.image_base64,
        image_url=image_uploaded,
        createdAt=time_stamp,
    )
    firebase_logger.log(log_data)  # Log to Firebase
    # -- Send notification to LINE
    line_notify_client.send(
        line_notify_client.template(
            name=response.rfid,
            success=response.success,
            timestamp=log_data.createdAt.isoformat(),  # Convert datetime to ISO 8601
            image_url=image_uploaded,  # Send the uploaded image URL if available
        ),
    )
    return response


@post(
    "/api/v1/create",
    tags=["internal"],
    include_in_schema=True,
    status_code=200,
    security=[{"x-api-key": []}],
    content_media_type="application/json",
    content_encoding="utf-8",
    description="Perform face recognition on the input image",
)
async def create(
    request: Request,
    data: schemas.UserCreationRequest,
) -> schemas.DefaultResponse:
    check_api_key(request.headers)  # Raises HTTPException if API key is invalid
    return await controller.add_face_to_database(data)


app = Litestar(
    route_handlers=[root, inference, create],
    exception_handlers={HTTPException: app_exception_handler},
    openapi_config=OpenAPIConfig(
        title="Face Recognition API - CedtEmbed",
        description="API for face recognition",
        version="0.1.0",
        tags=[
            Tag(
                name="internal",
                description="Internal API",
            )
        ],
        components=Components(
            security_schemes={
                "x-api-key": SecurityScheme(
                    type="apiKey",
                    name="x-api-key",
                    security_scheme_in="header",
                    description="API Key Header",
                )
            }
        ),
        render_plugins=[SwaggerRenderPlugin()],
        path="/docs/swagger",
    ),
    debug=os.environ.get("DEBUG", False),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
