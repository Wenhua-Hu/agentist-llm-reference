#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${API_KEY:?required}" "${MODEL_ID:?required}" "${MODEL_REVISION:?required}" "${BENCH_PROFILE:?required}"
export HOST="${HOST:-127.0.0.1}" PORT="${PORT:-8000}" HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
[[ "$BENCH_PROFILE" == baseline || "$BENCH_PROFILE" == optimized ]] || exit 2
[[ "$HOST" == 127.0.0.1 ]] || { echo 'Only loopback binding is supported'; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || exit 2
nvidia-smi -L
if ! command -v docker >/dev/null; then
  sudo -n apt-get update -qq
  sudo -n apt-get install -y -qq docker.io
  sudo -n systemctl enable --now docker
fi
sudo -n docker info >/dev/null
if ! sudo -n docker info --format '{{json .Runtimes}}' | python3 -c 'import json,sys;sys.exit(0 if "nvidia" in json.load(sys.stdin) else 1)'; then
  # Dedicated, disposable VM only. Do not restart Docker on a shared host.
  [[ -z "$(sudo -n docker ps -q)" ]] || { echo 'Cannot configure runtime while other containers run'; exit 1; }
  if ! command -v nvidia-ctk >/dev/null; then
    sudo -n apt-get update -qq
    sudo -n apt-get install -y -qq ca-certificates curl gnupg2
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo -n gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo -n tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo -n apt-get update -qq
    sudo -n apt-get install -y -qq nvidia-container-toolkit=1.20.1-1 \
      nvidia-container-toolkit-base=1.20.1-1 libnvidia-container-tools=1.20.1-1 libnvidia-container1=1.20.1-1
  fi
  sudo -n nvidia-ctk runtime configure --runtime=docker
  sudo -n systemctl restart docker
fi
mkdir -p .runtime "$HF_HOME"
chmod 700 .runtime
exec 9>.runtime/setup.lock
flock -w 110 9
export SOURCE_COMMIT="$(git rev-parse HEAD)"
fingerprint=$(python3 - <<'PY'
import hashlib,os
from pathlib import Path
h=hashlib.sha256()
for name in ['SOURCE_COMMIT','HOST','PORT','API_KEY','MODEL_ID','MODEL_REVISION','BENCH_PROFILE','HF_HOME']:
 h.update((name+'='+os.environ[name]+'\n').encode())
for p in sorted(Path('configs').glob('*.yaml')):h.update(p.read_bytes())
print(h.hexdigest())
PY
)
name=agentist-llm-reference
owner=$(sudo -n docker inspect -f '{{index .Config.Labels "agentist.project"}}' "$name" 2>/dev/null || true)
if [[ -n "$owner" && "$owner" != reference-v1 ]]; then
  echo 'Container name belongs to another project'; exit 1
fi
old=$(sudo -n docker inspect -f '{{index .Config.Labels "agentist.fingerprint"}}' "$name" 2>/dev/null || true)
if [[ "$old" == "$fingerprint" ]] && curl -fsS --max-time 3 "http://$HOST:$PORT/healthz" >/dev/null; then
  echo 'Same configuration already ready (idempotent)'; exit 0
fi
if [[ "$owner" == reference-v1 ]]; then
  sudo -n docker stop -t 25 "$name" >/dev/null || true
  sudo -n docker rm "$name" >/dev/null
fi
build_id=$(sha256sum Dockerfile requirements.lock | sha256sum | cut -c1-16)
image="agentist-llm-reference:$build_id"
if ! sudo -n docker image inspect "$image" >/dev/null 2>&1; then
  sudo -n docker build --progress=plain -t "$image" .
fi
# Secret is read from the inherited environment; never passed as a literal CLI argument.
sudo -n --preserve-env=API_KEY docker run -d --name "$name" --gpus all --network host \
  --shm-size 4g --label agentist.project=reference-v1 --label "agentist.fingerprint=$fingerprint" \
  -v "$PWD:/app:ro" -v "$PWD/.runtime:/runtime" -v "$HF_HOME:/hf" \
  -e API_KEY -e "HOST=$HOST" -e "PORT=$PORT" -e "MODEL_ID=$MODEL_ID" \
  -e "MODEL_REVISION=$MODEL_REVISION" -e "BENCH_PROFILE=$BENCH_PROFILE" \
  -e "SOURCE_COMMIT=$SOURCE_COMMIT" -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 \
  "$image" >/dev/null
printf '%s\n' "$image" > .runtime/image
for attempt in $(seq 1 150); do
  if curl -fsS --max-time 2 "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
    echo "Reference service ready: profile=$BENCH_PROFILE commit=$SOURCE_COMMIT"; exit 0
  fi
  if [[ "$(sudo -n docker inspect -f '{{.State.Running}}' "$name")" != true ]]; then break; fi
  sleep 1
done
echo 'Service did not become ready; engine/gateway diagnostics:'
tail -n 60 .runtime/engine.log .runtime/gateway.log || true
sudo -n docker stop -t 20 "$name" >/dev/null || true
exit 1
