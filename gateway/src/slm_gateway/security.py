"""Authentication and security dependencies for the SLM Gateway."""

from typing import Optional
from fastapi import Header, HTTPException, status

from .config import settings
from .schemas import OpenAIErrorResponse, OpenAIError


async def verify_api_key(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> None:
    """Verify Bearer token only if GATEWAY_API_KEY is configured.

    If GATEWAY_API_KEY is None or empty, authentication is disabled and
    all requests pass through.
    """
    if not settings.GATEWAY_API_KEY:
        return

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Missing Authorization header. Provide 'Authorization: Bearer <key>'",
                    "type": "authentication_error",
                    "code": 401,
                }
            },
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != settings.GATEWAY_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Incorrect API key provided.",
                    "type": "authentication_error",
                    "code": 401,
                }
            },
        )
