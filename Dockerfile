# Slim container image for the Nahual LSM web demo.
#
# Python 3.9 matches the local environment that trained and pickled the
# scikit-learn models, avoiding any cross-version unpickling issues.
FROM python:3.9-slim

WORKDIR /app

# Install the server-only dependencies first so this layer is cached
# across rebuilds when only application code changes.
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy the inference package, the FastAPI server, and the trained models.
# main.py / collect.py / train.py are intentionally excluded — they are the
# desktop OpenCV tools and are not part of the web demo.
COPY nahual/ ./nahual/
COPY server/ ./server/
COPY models/ ./models/

# Render injects the port to listen on via $PORT; default to 8000 so the
# image also runs locally with `docker run -p 8000:8000 nahual`.
ENV PORT=8000

# Shell form so ${PORT} is expanded at container start.
CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT}
