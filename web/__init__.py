"""Web package for the Nahual LSM gesture browser demo.

Marks ``web/`` as an importable package so the FastAPI application can be
referenced as ``web.app:app`` (e.g. by ``uvicorn`` in the Render/Docker
deployment). The desktop tools do not import this package; it exists solely
to host the thin browser-facing server in ``web/app.py``.
"""
