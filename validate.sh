#!/usr/bin/env bash
# =============================================================================
# validate.sh — Pre-submission validation for the Ad Platform OpenEnv project
#
# Usage:
#   ./validate.sh                          # uses defaults below
#   HF_SPACE_URL=https://... ./validate.sh # test against a live HF Space
#
# Checks:
#   Step 1/4 — Docker build succeeds
#   Step 2/4 — Server starts and /reset responds 200
#   Step 3/4 — /step responds correctly with obs_history
#   Step 4/4 — openenv validate passes
# =============================================================================

set -euo pipefail

# --------------- CONFIG ---------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="ad-platform-validate"
CONTAINER_NAME="ad-platform-validate-run"
SERVER_PORT=8000
LOCAL_URL="http://localhost:${SERVER_PORT}"
HF_SPACE_URL="${HF_SPACE_URL:-}"         # optional: set to test a live HF Space
DOCKER_BUILD_TIMEOUT=120                 # seconds
SERVER_STARTUP_WAIT=5                    # seconds to wait for server to be ready
TASK="${TASK:-auction}"                  # task to validate against

# --------------- COLOURS ---------------
BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m"

PASS_COUNT=0
FAIL_COUNT=0
CLEANUP_CONTAINER=false

# --------------- HELPERS ---------------
log()  { printf "  %b\n" "$*"; }
pass() { printf "  ${GREEN}${BOLD}[PASS]${NC} %b\n" "$*"; ((PASS_COUNT++)) || true; }
fail() { printf "  ${RED}${BOLD}[FAIL]${NC} %b\n" "$*"; ((FAIL_COUNT++)) || true; }
hint() { printf "  ${YELLOW}      hint:${NC} %b\n" "$*"; }
warn() { printf "  ${YELLOW}[WARN]${NC} %b\n" "$*"; }

stop_at() {
  cleanup
  printf "\n${RED}${BOLD}Stopped at %s — fix the issue above and re-run.${NC}\n\n" "$1"
  exit 1
}

cleanup() {
  if [ "$CLEANUP_CONTAINER" = true ]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_with_timeout() {
  local timeout=$1; shift
  if command -v timeout &>/dev/null; then
    timeout "$timeout" "$@"
  else
    # macOS fallback
    perl -e "alarm $timeout; exec @ARGV" "$@"
  fi
}

# --------------- HEADER ---------------
printf "\n${BOLD}=================================================${NC}\n"
printf "${BOLD}  Ad Platform — OpenEnv Pre-Submission Validator${NC}\n"
printf "${BOLD}=================================================${NC}\n"
printf "  Repo:   %s\n" "$REPO_DIR"
printf "  Task:   %s\n" "$TASK"
printf "  Image:  %s\n" "$IMAGE_NAME"
printf "\n"

# =============================================================================
# STEP 1 — Docker build
# =============================================================================
printf "${BOLD}Step 1/4: Docker build${NC}\n"

if ! command -v docker &>/dev/null; then
  fail "docker not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 1"
fi

if [ ! -f "$REPO_DIR/Dockerfile" ]; then
  fail "Dockerfile not found in repo root: $REPO_DIR"
  stop_at "Step 1"
fi

log "Building image '${IMAGE_NAME}' from $REPO_DIR ..."

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" \
  docker build -t "$IMAGE_NAME" "$REPO_DIR" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -30
  stop_at "Step 1"
fi

# =============================================================================
# STEP 2 — Server starts and /reset returns 200
# =============================================================================
printf "\n${BOLD}Step 2/4: Server startup + /reset${NC}\n"

# Clean up any leftover container from a previous run
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

log "Starting container on port ${SERVER_PORT} with TASK=${TASK} ..."
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "${SERVER_PORT}:${SERVER_PORT}" \
  -e "TASK=${TASK}" \
  "$IMAGE_NAME" >/dev/null
CLEANUP_CONTAINER=true

log "Waiting ${SERVER_STARTUP_WAIT}s for server to be ready ..."
sleep "$SERVER_STARTUP_WAIT"

# POST /reset
RESET_OUTPUT=$(mktemp)
RESET_HTTP=$(curl -s -o "$RESET_OUTPUT" -w "%{http_code}" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{}' \
  "${LOCAL_URL}/reset" \
  --max-time 15 2>/dev/null || echo "000")

if [ "$RESET_HTTP" = "200" ]; then
  pass "/reset returned HTTP 200"
else
  fail "/reset returned HTTP ${RESET_HTTP} (expected 200)"
  log "Response body:"
  cat "$RESET_OUTPUT" 2>/dev/null || true
  hint "Check container logs: docker logs ${CONTAINER_NAME}"
  stop_at "Step 2"
fi

# Validate response has expected observation fields
RESET_BODY=$(cat "$RESET_OUTPUT")

check_field() {
  local field="$1"
  if echo "$RESET_BODY" | grep -q "\"${field}\""; then
    pass "  /reset observation contains '${field}'"
  else
    fail "  /reset observation missing '${field}'"
    ((FAIL_COUNT++)) || true
  fi
}

check_field "remaining_budget"
check_field "campaign_performance"
check_field "obs_history"
check_field "step"
check_field "done"

# =============================================================================
# STEP 3 — /step returns correct response with obs_history populated
# =============================================================================
printf "\n${BOLD}Step 3/4: /step response and obs_history${NC}\n"

# Build a valid action for the selected task
if [ "$TASK" = "budget" ]; then
  ACTION='{"allocations": [50.0, 30.0, 20.0], "bids": []}'
else
  ACTION='{"allocations": [50.0, 30.0, 20.0], "bids": [0.65, 0.50, 0.35]}'
fi

STEP_OUTPUT=$(mktemp)
STEP_HTTP=$(curl -s -o "$STEP_OUTPUT" -w "%{http_code}" \
  -X POST \
  -H "Content-Type: application/json" \
  -d "{\"action\": ${ACTION}}" \
  "${LOCAL_URL}/step" \
  --max-time 15 2>/dev/null || echo "000")

if [ "$STEP_HTTP" = "200" ]; then
  pass "/step returned HTTP 200"
else
  fail "/step returned HTTP ${STEP_HTTP} (expected 200)"
  log "Response body:"
  cat "$STEP_OUTPUT" 2>/dev/null || true
  hint "Check container logs: docker logs ${CONTAINER_NAME}"
  stop_at "Step 3"
fi

STEP_BODY=$(cat "$STEP_OUTPUT")

check_step_field() {
  local field="$1"
  if echo "$STEP_BODY" | grep -q "\"${field}\""; then
    pass "  /step response contains '${field}'"
  else
    fail "  /step response missing '${field}'"
  fi
}

check_step_field "reward"
check_step_field "done"
check_step_field "remaining_budget"
check_step_field "competitor_bids"
check_step_field "obs_history"

# Check reward is a number (not null)
if echo "$STEP_BODY" | grep -qE '"reward":\s*-?[0-9]'; then
  pass "  reward is a non-null number"
else
  warn "  reward may be null or missing — check reward shaping"
fi

# Check obs_history has at least one entry after step 1
if echo "$STEP_BODY" | grep -qE '"obs_history":\s*\[.*"step"'; then
  pass "  obs_history is populated after step 1"
else
  warn "  obs_history appears empty after step 1 — check record_step() is called"
fi

# Stop container now that local checks are done
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
CLEANUP_CONTAINER=false

# =============================================================================
# STEP 4 — openenv validate
# =============================================================================
printf "\n${BOLD}Step 4/4: openenv validate${NC}\n"

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 4"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "$VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 4"
fi

# =============================================================================
# Optional Step — HF Space ping (only if HF_SPACE_URL is set)
# =============================================================================
if [ -n "$HF_SPACE_URL" ]; then
  printf "\n${BOLD}Bonus: HF Space ping${NC} (${HF_SPACE_URL})\n"
  HF_HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{}' \
    "${HF_SPACE_URL}/reset" \
    --max-time 30 2>/dev/null || echo "000")

  if [ "$HF_HTTP" = "200" ]; then
    pass "HF Space is live and /reset returned 200"
  elif [ "$HF_HTTP" = "000" ]; then
    fail "HF Space not reachable (connection failed or timed out)"
    hint "Check that the Space is running and the URL is correct"
  else
    fail "HF Space /reset returned HTTP ${HF_HTTP} (expected 200)"
  fi
fi

# =============================================================================
# SUMMARY
# =============================================================================
TOTAL=$((PASS_COUNT + FAIL_COUNT))
printf "\n${BOLD}=================================================${NC}\n"
if [ "$FAIL_COUNT" -eq 0 ]; then
  printf "${GREEN}${BOLD}  All checks passed! (%d/%d)${NC}\n" "$PASS_COUNT" "$TOTAL"
  printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
else
  printf "${RED}${BOLD}  %d check(s) failed out of %d.${NC}\n" "$FAIL_COUNT" "$TOTAL"
  printf "${RED}${BOLD}  Fix the failures above and re-run.${NC}\n"
fi
printf "${BOLD}=================================================${NC}\n\n"

[ "$FAIL_COUNT" -eq 0 ] && exit 0 || exit 1
