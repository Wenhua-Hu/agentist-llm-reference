#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
case_name= out=
while (($#)); do
  case "$1" in
    --case) case_name="$2"; shift 2;;
    --out) out="$2"; shift 2;;
    *) echo 'Unknown argument'; exit 2;;
  esac
done
case "$case_name" in
  kv) module=kv_cache;; prefix) module=prefix_cache;; attention) module=flash_attention;;
  *) echo '--case kv|prefix|attention required'; exit 2;;
esac
[[ "$out" == results/* && "$out" != *..* ]] || { echo '--out must be under results/'; exit 2; }
[[ -f .runtime/image || -f .runtime/runtime.json ]] || { echo 'Run setup.sh once to install pinned environment'; exit 1; }
# A principle experiment has its own pinned identity; never relabel a service
# model as a different experiment model after accepting caller configuration.
if [[ "$case_name" == kv ]]; then
  expected_model=Qwen/Qwen2.5-1.5B-Instruct
  expected_revision=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
  [[ "${KV_DTYPE-float32}" == float32 ]] || { echo 'KV experiment requires float32'; exit 2; }
  export KV_DTYPE=float32
else
  expected_model=Qwen/Qwen2.5-7B-Instruct
  expected_revision=a09a35458c702b33eeacc393d103063234e8bc28
fi
[[ "${MODEL_ID-$expected_model}" == "$expected_model" && "${MODEL_REVISION-$expected_revision}" == "$expected_revision" ]] || { echo 'Experiment model identity differs from the pinned case contract'; exit 2; }
export MODEL_ID="${MODEL_ID-$expected_model}"
export MODEL_REVISION="${MODEL_REVISION-$expected_revision}"
if [[ -f .runtime/runtime.json ]]; then
  # A fresh SSH shell does not inherit setup.sh's environment. Reuse its cache,
  # while allowing an explicit caller override. JSON values remain literal data.
  if [[ -z "${HF_HOME:-}" ]]; then
    HF_HOME=$(python3 -c 'import json,os; r=json.load(open(".runtime/runtime.json")); v=r.get("hf_home"); print(v if isinstance(v,str) and v else os.path.join(os.path.expanduser("~"),".cache/huggingface"))')
  fi
  export HF_HOME
  # Stop our managed processes, never unrelated processes in the sandbox.
  bash setup.sh --stop
  exec 9>.runtime/setup.lock; flock -w 10 9
  mkdir -p "$out"
  engine_python=$(python3 -c 'import json; print(json.load(open(".runtime/runtime.json"))["engine_python"])')
  engine_bin=$(python3 -c 'import json,pathlib; r=json.load(open(".runtime/runtime.json")); print(r.get("engine_bin") or pathlib.Path(r["engine_python"]).absolute().parent)')
  gateway_bin=$(python3 -c 'import json,pathlib; r=json.load(open(".runtime/runtime.json")); print(r.get("gateway_bin") or pathlib.Path(r.get("gateway_python",r["engine_python"])).absolute().parent)')
  export CUDA_HOME=$(python3 -c 'import json,os; r=json.load(open(".runtime/runtime.json")); print(r.get("cuda_home") or os.environ.get("CUDA_HOME","/usr/local/cuda"))')
  export PATH="$engine_bin:$gateway_bin:$CUDA_HOME/bin:$PATH"
  export SGLANG_PYTHON="$engine_python" SOURCE_COMMIT="$(git rev-parse HEAD)"
  export HF_HUB_DISABLE_XET=1 KV_DTYPE="${KV_DTYPE-float32}"
  "$engine_python" -m "experiments.$module" --out "$out"
  echo 'Experiment finished; run setup.sh explicitly to restart serving.'
  exit 0
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
image=$(cat .runtime/image)
exec 9>.runtime/setup.lock; flock -w 10 9
owner=$(sudo -n docker inspect -f '{{index .Config.Labels "agentist.project"}}' agentist-llm-reference 2>/dev/null || true)
[[ "$owner" == reference-v1 ]] || { echo 'Expected own service container'; exit 1; }
# This is an explicit experiment: stop the service before loading a second model.
sudo -n docker stop -t 25 agentist-llm-reference >/dev/null
mkdir -p "$out"
sudo -n docker run --rm --gpus all --network host --shm-size 4g --entrypoint python3 \
  -v "$PWD:/app:ro" -v "$PWD/results:/app/results" -v "$HF_HOME:/hf" \
  -e "MODEL_ID=$MODEL_ID" -e "MODEL_REVISION=$MODEL_REVISION" -e HF_HOME=/hf \
  -e "SOURCE_COMMIT=$(git rev-parse HEAD)" -e HF_HUB_DISABLE_XET=1 -e "KV_DTYPE=${KV_DTYPE-float32}" \
  "$image" -m "experiments.$module" --out "$out"
echo 'Experiment finished; run setup.sh explicitly to restart serving.'
