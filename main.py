"""
Application entry point.

Two ways to run this:

    python main.py                  # uses the settings below directly
    uvicorn main:app --reload       # standard uvicorn CLI invocation

`app` is exposed at module level (not just inside the __main__ guard)
specifically so the second form works -- `uvicorn main:app` imports
this module and looks for a module-level ASGI object named `app`.

The `if __name__ == "__main__":` guard below means the `uvicorn.run(...)`
call only executes when this file is run directly (`python main.py`),
not when `main` is merely imported (e.g. by uvicorn's own import
machinery, or by tests).
"""
from __future__ import annotations

import uvicorn

from app.config import settings
from app.interface_adapters.api.app_factory import create_app

app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )