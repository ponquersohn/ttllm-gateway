#!/bin/sh

TTLLM_EXIT_ON_ERROR="${TTLLM_EXIT_ON_ERROR:-true}"

alembic upgrade head && python -m ttllm.handlers.ecs_entrypoint
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ] && [ "$TTLLM_EXIT_ON_ERROR" != "true" ]; then
    echo "Process exited with code $EXIT_CODE — TTLLM_EXIT_ON_ERROR is not true, keeping container alive for debugging."
    while true; do sleep 3600; done
fi

exit $EXIT_CODE
