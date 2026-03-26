"""ECS / local development entrypoint via Uvicorn."""

from ttllm.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ttllm.handlers.ecs_entrypoint:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
    )
