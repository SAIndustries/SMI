#!/usr/bin/env bash
# validate-submission.sh — OpenEnv Submission Validator
# Usage: ./validate-submission.sh <hf_space_url> [repo_dir]
set -uo pipefail
DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then RED='\033[0;31m';GREEN='\033[0;32m';YELLOW='\033[1;33m';BOLD='\033[1m';NC='\033[0m'
else RED='';GREEN='';YELLOW='';BOLD='';NC='';fi
run_with_timeout(){local s="$1";shift;if command -v timeout &>/dev/null;then timeout "$s" "$@";elif command -v gtimeout &>/dev/null;then gtimeout "$s" "$@";else "$@"&local p=$!;(sleep "$s"&&kill "$p" 2>/dev/null)&local w=$!;wait "$p" 2>/dev/null;local r=$?;kill "$w" 2>/dev/null;wait "$w" 2>/dev/null;return $r;fi;}
portable_mktemp(){mktemp "${TMPDIR:-/tmp}/${1:-validate}-XXXXXX" 2>/dev/null||mktemp;}
CLEANUP_FILES=();cleanup(){rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}";}
trap cleanup EXIT
PING_URL="${1:-}";REPO_DIR="${2:-.}"
if [ -z "$PING_URL" ];then printf "Usage: %s <ping_url> [repo_dir]\n" "$0";exit 1;fi
if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null&&pwd)";then printf "Error: directory '%s' not found\n" "${2:-.}";exit 1;fi
PING_URL="${PING_URL%/}";PASS=0
log(){printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*";}
pass(){log "${GREEN}PASSED${NC} -- $1";PASS=$((PASS+1));}
fail(){log "${RED}FAILED${NC} -- $1";}
hint(){printf "  ${YELLOW}Hint:${NC} %b\n" "$1";}
stop_at(){printf "\n${RED}${BOLD}Stopped at %s.${NC} Fix above before continuing.\n" "$1";exit 1;}
printf "\n${BOLD}========================================${NC}\n${BOLD}  OpenEnv Submission Validator${NC}\n${BOLD}========================================${NC}\n"
log "Repo: $REPO_DIR";log "URL:  $PING_URL";printf "\n"
log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."
CURL_OUTPUT=$(portable_mktemp "validate-curl");CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT"||printf "000")
if [ "$HTTP_CODE"="200" ];then pass "HF Space live, /reset returns 200"
elif [ "$HTTP_CODE"="000" ];then fail "Not reachable";hint "Try: curl -X POST $PING_URL/reset";stop_at "Step 1"
else fail "/reset returned HTTP $HTTP_CODE";stop_at "Step 1";fi
log "${BOLD}Step 2/3: docker build${NC} ..."
if ! command -v docker &>/dev/null;then fail "docker not found";hint "https://docs.docker.com/get-docker/";stop_at "Step 2";fi
if [ -f "$REPO_DIR/Dockerfile" ];then DC="$REPO_DIR";elif [ -f "$REPO_DIR/server/Dockerfile" ];then DC="$REPO_DIR/server";else fail "No Dockerfile";stop_at "Step 2";fi
BUILD_OK=false;BO=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DC" 2>&1)&&BUILD_OK=true
if [ "$BUILD_OK"=true ];then pass "Docker build succeeded";else fail "Docker build failed";printf "%s\n" "$BO"|tail -20;stop_at "Step 2";fi
log "${BOLD}Step 3/3: openenv validate${NC} ..."
if ! command -v openenv &>/dev/null;then fail "openenv not found";hint "pip install openenv-core";stop_at "Step 3";fi
VO_OK=false;VO=$(cd "$REPO_DIR"&&openenv validate 2>&1)&&VO_OK=true
if [ "$VO_OK"=true ];then pass "openenv validate passed";[ -n "$VO" ]&&log "  $VO"
else fail "openenv validate failed";printf "%s\n" "$VO";stop_at "Step 3";fi
printf "\n${BOLD}========================================${NC}\n${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n${GREEN}${BOLD}  Your submission is ready.${NC}\n${BOLD}========================================${NC}\n\n"
exit 0
