#!/usr/bin/env bash
# Bring a Colab VM, fresh or half set up, to: repo current, vLLM installed, weights cached, model served,
# tunnel up. Safe to rerun after a crash or a reconnect: each step skips work that is already done.
#
# Needs:   REPO_URL, VLLM_API_KEY. GITHUB_TOKEN for a private repo.
# Tunnel:  ngrok when NGROK_AUTHTOKEN is set (NGROK_DOMAIN pins the URL), else a Cloudflare quick tunnel.
# Engine:  the vLLM wheel at VLLM_VERSION, or VLLM_GIT=<url>@<ref> to install a fork with its Python editable.
# Prints the public URL last and writes it to $STATE/url.
set -euo pipefail

: "${REPO_URL:?set REPO_URL}" "${VLLM_API_KEY:?set VLLM_API_KEY}"
MODEL=${MODEL:-Qwen/Qwen3-4B}
VLLM_VERSION=${VLLM_VERSION:-0.31.0}
VLLM_ARGS=${VLLM_ARGS:-}
PORT=${PORT:-8000}
WORK=${WORK:-/content}
STATE=$WORK/state
export HF_HOME=${HF_HOME:-$WORK/hf}
mkdir -p "$STATE" "$WORK/bin"
export PATH=$WORK/bin:$PATH

say() { printf '\n== %s\n' "$*"; }

git_auth() {
  # shellcheck disable=SC2016  # $GITHUB_TOKEN expands inside git's helper, not here.
  # The token reaches git through a credential helper so it is never written into .git/config.
  git -c credential.helper= -c credential.helper='!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' "$@"
}

repo() {
  local dir=$WORK/spelunking
  say "repo"
  if [ -d "$dir/.git" ]; then git_auth -C "$dir" pull --ff-only; else git_auth clone "$REPO_URL" "$dir"; fi
  git -C "$dir" log --oneline -1
}

engine() {
  say "vLLM"
  if [ -n "${VLLM_GIT:-}" ]; then
    local url=${VLLM_GIT%@*} ref=${VLLM_GIT##*@} dir=$WORK/vllm-core
    [ -d "$dir/.git" ] || git_auth clone --filter=blob:none "$url" "$dir"
    git_auth -C "$dir" fetch origin "$ref"
    git -C "$dir" checkout --detach FETCH_HEAD
    # Python edits only: the compiled kernels come from the matching prebuilt wheel, so this takes minutes.
    (cd "$dir" && VLLM_USE_PRECOMPILED=1 uv pip install --system -e .)
  elif python -c "import vllm, sys; sys.exit(vllm.__version__ != '$VLLM_VERSION')" 2>/dev/null; then
    echo "vllm $VLLM_VERSION already installed"
  else
    uv pip install --system "vllm==$VLLM_VERSION"
  fi
}

weights() {
  say "weights: $MODEL"
  python -c "from huggingface_hub import snapshot_download as d; print(d('$MODEL'))"
}

healthy() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; }

serve() {
  say "server"
  if healthy; then echo "already serving on :$PORT"; return; fi
  local args; read -ra args <<<"$VLLM_ARGS"
  # Turing (T4) has no bf16; vLLM refuses the model's bf16 default there.
  local cap; cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  if [ "${cap%%.*}" -lt 8 ]; then args+=(--dtype half); fi
  nohup vllm serve "$MODEL" --port "$PORT" --api-key "$VLLM_API_KEY" "${args[@]}" >"$STATE/server.log" 2>&1 &
  echo $! >"$STATE/server.pid"
  for _ in $(seq 900); do
    healthy && { echo "healthy after ${SECONDS}s total"; return; }
    kill -0 "$(cat "$STATE/server.pid")" 2>/dev/null || { tail -40 "$STATE/server.log"; echo "server died"; exit 1; }
    sleep 1
  done
  tail -40 "$STATE/server.log"; echo "server not healthy after 15 min"; exit 1
}

tunnel() {
  say "tunnel"
  if [ -f "$STATE/tunnel.pid" ] && kill -0 "$(cat "$STATE/tunnel.pid")" 2>/dev/null; then
    echo "already up"
  elif [ -n "${NGROK_AUTHTOKEN:-}" ]; then
    command -v ngrok >/dev/null ||
      curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar xz -C "$WORK/bin"
    ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null
    nohup ngrok http "$PORT" ${NGROK_DOMAIN:+--url "$NGROK_DOMAIN"} --log stdout >"$STATE/tunnel.log" 2>&1 &
    echo $! >"$STATE/tunnel.pid"
    for _ in $(seq 30); do
      url=$(curl -sf http://127.0.0.1:4040/api/tunnels | python -c "import json,sys; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null) && break
      sleep 1
    done
    echo "${url:?ngrok gave no URL; see $STATE/tunnel.log}" >"$STATE/url"
  else
    command -v cloudflared >/dev/null || {
      curl -sSLo "$WORK/bin/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
      chmod +x "$WORK/bin/cloudflared"
    }
    nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$STATE/tunnel.log" 2>&1 &
    echo $! >"$STATE/tunnel.pid"
    for _ in $(seq 30); do
      url=$(grep -om1 'https://[a-z0-9-]*\.trycloudflare\.com' "$STATE/tunnel.log") && break
      sleep 1
    done
    echo "${url:?cloudflared gave no URL; see $STATE/tunnel.log}" >"$STATE/url"
  fi
  say "URL: $(cat "$STATE/url")"
}

repo
engine
weights
serve
tunnel
