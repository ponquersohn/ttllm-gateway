"""ECS / local development entrypoint via Uvicorn."""

from ttllm.api.app import create_app
from ttllm.config import settings

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ttllm.handlers.ecs_entrypoint:app",
        host="0.0.0.0",
        port=settings.engine.listen_port,
        workers=4,
    )
