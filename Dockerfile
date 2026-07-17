# Slim container image for the Nahual LSM web demo.
#
# Python 3.9 matches the local environment that trained and pickled the
# scikit-learn models, avoiding any cross-version unpickling issues.
FROM python:3.9-slim

WORKDIR /app

# Install the server-only dependencies first so this layer is cached
# across rebuilds when only application code changes. These are intentionally
# slim: the server never imports mediapipe or opencv (MediaPipe runs in the
# browser and only hand landmarks are streamed back).
COPY web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared inference package, the FastAPI server, and the trained
# models. The desktop OpenCV tools (main.py / collect.py / train.py) and the
# raw training data are intentionally excluded via .dockerignore — they are
# not part of the web demo.
COPY nahual/ ./nahual/
COPY web/ ./web/
COPY models/ ./models/

# Render injects the port to listen on via $PORT; default to 8000 so the
# image also runs locally with `docker run -p 8000:8000 nahual`.
ENV PORT=8000

# Shell form so ${PORT} is expanded at container start. web/app.py resolves
# the project root relative to itself, so running the module from /app finds
# the models/ and web/static/ directories copied above.
CMD uvicorn web.app:app --host 0.0.0.0 --port ${PORT}
