"""HTTP Basic Auth dependency. Single user, password from env."""
from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .config import get_settings

_security = HTTPBasic(realm="LectureMind")


def require_auth(
    creds: Annotated[HTTPBasicCredentials, Depends(_security)],
) -> str:
    settings = get_settings()
    user_ok = secrets.compare_digest(
        creds.username.encode("utf-8"),
        settings.basic_auth_user.encode("utf-8"),
    )
    pwd_ok = secrets.compare_digest(
        creds.password.encode("utf-8"),
        settings.basic_auth_password.encode("utf-8"),
    )
    if not (user_ok and pwd_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": 'Basic realm="LectureMind"'},
        )
    return creds.username
