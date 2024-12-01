from dataclasses import dataclass
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
from controller import FaceRecognitionController
from libs.firebase_logger import FirebaseAuthenticationLogs
from libs.auth import check_api_key
import models.schemas as schemas
from libs.exception_handler import app_exception_handler
import os

load_dotenv()


@get("/", include_in_schema=True)
async def root() -> str:
    return "OK!"


controller = FaceRecognitionController()
firebase_logger = FirebaseAuthenticationLogs()


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
    firebase_logger.log(
        schemas.FirebaseLog(
            **response.model_dump(),
        )
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
    response = await controller.add_face_to_database(data)
    firebase_logger.log()
    return


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
