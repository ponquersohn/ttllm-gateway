#!/usr/bin/env bash
#
# Build and run the Claude Code sandbox against the TTLLM gateway.
#
# Builds the image from Dockerfile_claude (context = repo root, so the corporate
# CA certs in work/cacerts/ get copied in), then drops you into an interactive
# bash shell with work/ mounted at /work and `claude` on PATH.
#
# Rather than baking a long-lived API key into work/.env, this script mints a
# fresh gateway token via `ttllm tokens create` (labelled "test docker token"),
# injects it as ANTHROPIC_API_KEY for the container, and revokes it again as
# soon as the container exits. The base URL and model settings still come from
# work/.env (ANTHROPIC_BASE_URL, ANTHROPIC_MODEL, ...); the -e override below
# wins over the API key in that file. Requires an existing `ttllm login`
# session.
#
#   ./test_claude_docker.sh                 # interactive shell, then run `claude`
#   ./test_claude_docker.sh claude          # launch claude directly
#   ./test_claude_docker.sh claude -p "hi"  # one-shot prompt
#
# --network host lets the localhost gateway URL in .env reach the gateway
# running on the host (works on Linux / WSL2).
set -euo pipefail

IMAGE="${TTLLM_CLAUDE_IMAGE:-ttllm-claude}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${REPO_DIR}/work"
ENV_FILE="${WORK_DIR}/.env"
TTLLM="${REPO_DIR}/.venv/bin/ttllm"
TOKEN_LABEL="test docker token"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "error: ${ENV_FILE} not found — create it with ANTHROPIC_BASE_URL / ANTHROPIC_MODEL" >&2
    exit 1
fi

if [[ ! -x "${TTLLM}" ]]; then
    echo "error: ${TTLLM} not found — set up the project venv first" >&2
    exit 1
fi

# Remove any leftover token with this label from a previous run (e.g. one that
# crashed before its cleanup). `me tokens` lists only the current user's tokens.
for stale_id in $("${TTLLM}" me tokens --json | "${REPO_DIR}/.venv/bin/python" -c '
import json, sys
label = sys.argv[1]
for t in json.load(sys.stdin):
    if t.get("label") == label:
        print(t["id"])
' "${TOKEN_LABEL}"); do
    echo "Removing leftover token ${stale_id}..."
    "${TTLLM}" tokens delete "${stale_id}" >/dev/null 2>&1 || \
        echo "warning: failed to remove leftover token ${stale_id}" >&2
done

# Mint a short-lived gateway token for this run. Requires an active session;
# `ttllm tokens create` errors out (and we exit) if you are not logged in.
echo "Creating gateway token '${TOKEN_LABEL}'..."
TOKEN_JSON="$("${TTLLM}" tokens create --label "${TOKEN_LABEL}" --json)"
TOKEN_ID="$(printf '%s' "${TOKEN_JSON}" | "${REPO_DIR}/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
TOKEN_VALUE="$(printf '%s' "${TOKEN_JSON}" | "${REPO_DIR}/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"

# Always revoke the token when the script exits, however it exits.
cleanup() {
    echo "Revoking gateway token ${TOKEN_ID}..."
    "${TTLLM}" tokens delete "${TOKEN_ID}" >/dev/null 2>&1 || \
        echo "warning: failed to revoke token ${TOKEN_ID}; revoke it manually" >&2
}
trap cleanup EXIT

# Always rebuild so cert/Dockerfile edits are picked up; layer caching keeps it fast.
echo "Building ${IMAGE}..."
docker build -t "${IMAGE}" -f "${REPO_DIR}/Dockerfile_claude" "${REPO_DIR}"

# Note: no `exec` here so the EXIT trap runs and revokes the token after the
# container exits. -e ANTHROPIC_API_KEY overrides whatever is in --env-file.
docker run --rm -it \
    --network host \
    --env-file "${ENV_FILE}" \
    -e ANTHROPIC_API_KEY="${TOKEN_VALUE}" \
    -v "${WORK_DIR}:/work" \
    -w /work \
    "${IMAGE}" "${@:-bash}"
