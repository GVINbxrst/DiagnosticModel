"""Test environment compatibility shims.

Currently retained:
 - httpx.AsyncClient(app=...) backward compatibility for tests using deprecated shortcut.

Removed (now handled by UniversalUUID type in models):
 - UUID coercion patches for SQLAlchemy (bind processors, session.get, engine listeners).
These were redundant after introducing `UniversalUUID` TypeDecorator and could mask real issues.
"""
from __future__ import annotations

import inspect
import sqlite3
from datetime import datetime

# Silence Python 3.12+ sqlite datetime adapter deprecation by registering explicit adapters/converters
try:  # pragma: no cover - environment dependent
    # Store datetime as ISO string
    sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
except Exception:
    pass

try:
    import httpx  # type: ignore
    from httpx import ASGITransport  # type: ignore
except Exception:  # pragma: no cover - httpx must exist for tests
    httpx = None  # type: ignore

if httpx is not None:  # pragma: no branch
    try:
        sig = inspect.signature(httpx.AsyncClient.__init__)
        if 'app' not in sig.parameters:
            _orig_init = httpx.AsyncClient.__init__  # type: ignore

            def _patched_init(self, *args, app=None, transport=None, base_url: str = '', **kwargs):  # type: ignore
                # If user passed app and no explicit transport, wrap with ASGITransport
                if app is not None and transport is None:
                    transport = ASGITransport(app=app)
                return _orig_init(self, *args, transport=transport, base_url=base_url, **kwargs)

            httpx.AsyncClient.__init__ = _patched_init  # type: ignore
    except Exception:
        pass

## All UUID monkeypatch logic removed; UniversalUUID covers cross-dialect behavior.
