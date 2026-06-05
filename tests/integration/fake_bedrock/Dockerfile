FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30"

# The fake server only needs its own package (encoder + app), imported as a
# top-level package inside the image.
COPY tests/integration/fake_bedrock/ ./fake_bedrock/

EXPOSE 9099

CMD ["uvicorn", "fake_bedrock.app:app", "--host", "0.0.0.0", "--port", "9099"]
