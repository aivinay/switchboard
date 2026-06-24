#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

pretty_json() {
  if command -v jq >/dev/null 2>&1; then
    jq .
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m json.tool
  else
    cat
  fi
}

post_route() {
  local title="$1"
  local prompt="$2"

  printf '\n%s\n' "================================================================================"
  printf '%s\n\n' "$title"
  curl -sS "$BASE_URL/personal/route" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"$prompt\", \"project\": \"demo\"}" | pretty_json
}

printf '\nSwitchboard demo\n'
printf 'Story: simple tasks stay local/mock, scarce premium tools are recommended only, '
printf 'and private prompts do not go to cloud providers by default.\n'

curl -sS "$BASE_URL/personal/health" | pretty_json

post_route "1. Simple summarisation avoids premium models" \
  "Summarise this customer support ticket in three bullets."

post_route "2. Coding/debugging prefers a local or mock coding-capable route" \
  "Debug this Python code: for customer in customers: print(customer_id)"

post_route "3. Complex planning recommends premium/manual help without calling it" \
  "Create a multi-step strategy for launching a local-first developer tool."

post_route "4. Private prompt stays local by default" \
  "This is private: summarise my personal medical notes without using cloud models."

printf '\n%s\n' "================================================================================"
printf 'Usage summary\n\n'
curl -sS "$BASE_URL/personal/usage" | pretty_json

printf '\n%s\n' "================================================================================"
printf 'Recent personal history\n\n'
curl -sS "$BASE_URL/personal/history?limit=4" | pretty_json
