import os
from litestar.datastructures import Headers
from litestar import exceptions
from hashlib import sha256


def check_api_key(header: Headers):
    API_KEY = sha256(os.environ["SALTED_API_KEY"].encode()).hexdigest()
    ENCODED = sha256(header.get("x-api-key", "").encode()).hexdigest()
    if ENCODED == API_KEY:
        return True

    raise exceptions.HTTPException(
        status_code=401,
        detail="Unauthorized",
        headers={"x-api-key": API_KEY},
    )
