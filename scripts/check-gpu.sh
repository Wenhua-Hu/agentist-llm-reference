#!/usr/bin/env bash
# Validate CUDA in the actual image before model startup. Bounded readiness retry,
# never an inference-test retry. No driver changes and no CPU fallback.
set -euo pipefail
image="${1:?container image required}"
probe_name="agentist-llm-gpu-probe-$$-$RANDOM"
cleanup(){ sudo -n docker rm -f "$probe_name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
for attempt in $(seq 1 6); do
  if sudo -n timeout --signal=TERM --kill-after=5s 25s docker run --rm --name "$probe_name" --gpus all --entrypoint python3 "$image" -c 'import subprocess,torch; subprocess.run(["nvidia-smi","-L"],check=True); assert torch.cuda.is_available(), "CUDA unavailable in container"; assert torch.ones(1,device="cuda").sum().item()==1; torch.cuda.synchronize()'; then
    echo 'Container NVML and CUDA allocation ready'; exit 0
  fi
  cleanup
  echo "Container GPU readiness probe failed ($attempt/6)"
  if [[ "$attempt" -lt 6 ]]; then sleep 5; fi
done
echo 'Container GPU remains unavailable; refusing to start an unhealthy model service'
exit 1
