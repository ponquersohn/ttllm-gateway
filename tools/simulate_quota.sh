#!/usr/bin/env bash
#
# simulate_quota.sh — exercise time-based token-quota enforcement end-to-end.
#
# Spins up a throwaway user + gateway token, applies a SHORT user-scoped limit
# (via window_seconds, so a real 5h window isn't required), then fires requests
# and observes the full lifecycle: consume -> breach (429) -> cooldown -> window
# reset -> access restored. Cleans up the user and limit on exit.
#
# Requires: a running gateway (e.g. docker-compose.integration.yml up), an admin
# account, at least one model the test user can be assigned, curl, python3.
#
# Usage:
#   tools/simulate_quota.sh [--url URL] [--cap TOKENS] [--window SECONDS]
#                           [--max-requests N] [--admin-email EMAIL]
#                           [--admin-password PW] [--model NAME]
#
# Defaults: --url http://localhost:8000  --cap 80  --window 15  --max-requests 10
#           --admin-email admin@localhost  --admin-password admin
#           --model <first model returned by /admin/models>
#
# Note: under debit-after-the-fact accounting, the request that crosses the cap
# is allowed to COMPLETE; the NEXT request is the one blocked (429). So a 429 may
# surface on the step-4 cooldown retry rather than inside the step-3 loop — both
# are correct enforcement.
#
set -uo pipefail

BASE="http://localhost:8000"
CAP=80
WINDOW=15
MAX_REQ=10
ADMIN_EMAIL="admin@localhost"
ADMIN_PASSWORD="admin"
MODEL_NAME=""

while [ $# -gt 0 ]; do
  case "$1" in
    --url)            BASE="$2"; shift 2 ;;
    --cap)            CAP="$2"; shift 2 ;;
    --window)         WINDOW="$2"; shift 2 ;;
    --max-requests)   MAX_REQ="$2"; shift 2 ;;
    --admin-email)    ADMIN_EMAIL="$2"; shift 2 ;;
    --admin-password) ADMIN_PASSWORD="$2"; shift 2 ;;
    --model)          MODEL_NAME="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

jqget() { python3 -c "import sys,json;print(json.load(sys.stdin)$1)" 2>/dev/null; }

# --- cleanup runs even on early exit ---
LIMIT_ID=""; USER_ID=""; ADMIN_AUTH=""
cleanup() {
  [ -n "$ADMIN_AUTH" ] || return
  [ -n "$LIMIT_ID" ] && curl -s -o /dev/null -X DELETE "$BASE/admin/usage-limits/$LIMIT_ID" -H "$ADMIN_AUTH"
  [ -n "$USER_ID" ]  && curl -s -o /dev/null -X DELETE "$BASE/admin/users/$USER_ID" -H "$ADMIN_AUTH"
  echo "  (cleaned up test limit + user)"
}
trap cleanup EXIT

echo "===== 0. Admin login ($BASE) ====="
ADMIN_TOKEN=$(curl -s -X POST "$BASE/auth/token" -H "Content-Type: application/json" \
  -d "{\"email\":\"$ADMIN_EMAIL\",\"password\":\"$ADMIN_PASSWORD\"}" | jqget "['access_token']")
if [ -z "$ADMIN_TOKEN" ]; then echo "  admin login failed — is the gateway up at $BASE?" >&2; exit 1; fi
ADMIN_AUTH="Authorization: Bearer $ADMIN_TOKEN"
echo "  ok"

if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME=$(curl -s "$BASE/admin/models?limit=5" -H "$ADMIN_AUTH" | jqget "['items'][0]['name']")
fi
MODEL_ID=$(curl -s "$BASE/admin/models?limit=50" -H "$ADMIN_AUTH" \
  | python3 -c "import sys,json;n='$MODEL_NAME';[print(m['id']) for m in json.load(sys.stdin)['items'] if m['name']==n]" 2>/dev/null | head -1)
if [ -z "$MODEL_ID" ]; then echo "  no model named '$MODEL_NAME' — create/assign one first" >&2; exit 1; fi
echo "  model: $MODEL_NAME"

echo "===== 1. Create throwaway user + assign model + mint gateway token ====="
EMAIL="quota-sim-$(date +%s)@example.com"
USER_ID=$(curl -s -X POST "$BASE/admin/users" -H "$ADMIN_AUTH" -H "Content-Type: application/json" \
  -d "{\"name\":\"Quota Sim\",\"email\":\"$EMAIL\",\"password\":\"QuotaSim1!\"}" | jqget "['id']")
curl -s -o /dev/null -X POST "$BASE/admin/users/$USER_ID/permissions" -H "$ADMIN_AUTH" \
  -H "Content-Type: application/json" -d '{"permissions":["llm.invoke"]}'
curl -s -o /dev/null -X POST "$BASE/admin/models/$MODEL_ID/assign" -H "$ADMIN_AUTH" \
  -H "Content-Type: application/json" -d "{\"user_ids\":[\"$USER_ID\"]}"
GW_TOKEN=$(curl -s -X POST "$BASE/admin/tokens" -H "$ADMIN_AUTH" -H "Content-Type: application/json" \
  -d "{\"user_id\":\"$USER_ID\",\"label\":\"quota-sim\",\"permissions\":[\"llm.invoke\"]}" | jqget "['access_token']")
echo "  user $USER_ID ready"

echo "===== 2. Apply limit: $CAP tokens / ${WINDOW}s window (user-scoped) ====="
LIMIT_ID=$(curl -s -X POST "$BASE/admin/usage-limits" -H "$ADMIN_AUTH" -H "Content-Type: application/json" \
  -d "{\"scope\":\"user\",\"user_id\":\"$USER_ID\",\"window_kind\":\"5h\",\"token_cap\":$CAP,\"window_seconds\":$WINDOW}" | jqget "['id']")
echo "  limit $LIMIT_ID"

invoke() {
  curl -s -o /tmp/qsim_body.json -w "%{http_code}" -X POST "$BASE/anthropic/v1/messages" \
    -H "x-api-key: $GW_TOKEN" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Tell me briefly about coffee.\"}],\"max_tokens\":64}"
}

echo "===== 3. Fire up to $MAX_REQ requests; expect 200s then a 429 ====="
BREACHED=0
for i in $(seq 1 "$MAX_REQ"); do
  CODE=$(invoke)
  if [ "$CODE" = "200" ]; then
    echo "  req $i: 200  usage=$(jqget "['usage']['input_tokens']" </tmp/qsim_body.json)in/$(jqget "['usage']['output_tokens']" </tmp/qsim_body.json)out"
  elif [ "$CODE" = "429" ]; then
    echo "  req $i: 429 <- QUOTA ENFORCED: $(jqget "['error']['message']" </tmp/qsim_body.json)"
    BREACHED=1; break
  else
    echo "  req $i: $CODE — $(head -c 160 /tmp/qsim_body.json)"; break
  fi
  sleep 1
done
[ "$BREACHED" = "1" ] || echo "  (cap not crossed inside the loop — the breaching request may be the last 200; checking cooldown next)"

echo "===== 4. Still blocked during cooldown? ====="
sleep 2
CODE=$(invoke)
echo "  retry: $CODE (expect 429)"
[ "$CODE" = "429" ] && BREACHED=1

echo "===== 5. Wait out the window, expect access restored ====="
echo "  sleeping $((WINDOW + 1))s..."
sleep $((WINDOW + 1))
echo "  post-window retry: $(invoke) (expect 200)"

echo "===== DONE ====="
