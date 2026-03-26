"""Local dev server — open this file and hit F5 to debug."""

import os

os.environ.setdefault("TTLLM_CONFIG_FILE", "config.yaml")
os.environ.setdefault("TTLLM_CONFIG_ENV", "dev")

import uvicorn

from ttllm.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
