from http.client import HTTPException
import traceback
from litestar import Request, Response


def app_exception_handler(request: Request, exc: Exception) -> Response:
    if isinstance(exc, HTTPException) and exc.status_code in {401, 403}:
        # Do not log stack traces for expected HTTP exceptions like Unauthorized
        return Response(
            content={
                "error": "Unauthorized" if exc.status_code == 401 else "Forbidden",
                "path": request.url.path,
                "detail": exc.detail,
                "status_code": exc.status_code,
            },
            status_code=exc.status_code,
        )
    else:
        # Log unexpected exceptions with stack trace
        traceback_str = traceback.format_exc()
        print(
            f"Unexpected exception for {request.method} {request.url}: {traceback_str}"
        )
        return Response(
            content={
                "error": "Internal Server Error",
                "path": request.url.path,
                "detail": "An unexpected error occurred",
                "status_code": 500,
            },
            status_code=500,
        )
