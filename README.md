# spelunking

Hands-on inference engineering on a MacBook Air M5 (8-core GPU, 16 GB, fanless), toward a masters-thesis
direction. Questions come from the research-gap search in `research/`; each is tested by a one-change experiment
against vLLM 0.31.0 with the vllm-metal plugin.

| Folder | What |
|---|---|
| `experiments/` | the experiments, their raw runs, the ledger, and the plain-language journal. Start at `experiments/README.md`. |
| `research/` | research-gap-dossier artifacts: source ledger, candidate cards, dossier, decisions. The candidates decide which experiments run. |

The engine lives outside this folder, at `~/dev/vllm-lab/`:

| Path | What | Branch |
|---|---|---|
| `vllm-core/` | vLLM engine (scheduler, KV block manager, spec decode) | `lab` off tag `v0.31.0` |
| `vllm-metal/` | Apple GPU plugin (attention kernels, spec decode, KV policy) | `lab` off upstream `main` @ 572322e |
| `.venv/` | Python 3.12, both clones installed editable; `setup.sh` rebuilds it from `requirements.lock` | |

The clone is named `vllm-core`, not `vllm`: a folder named `vllm` in the working directory shadows the installed
package and breaks vLLM's subprocesses. Experiment scripts find the clones through `VLLM_LAB`
(default `~/dev/vllm-lab`) to record their SHAs in every run.

## Environment

```sh
source ~/dev/vllm-lab/.venv/bin/activate
```

vLLM 0.31.0, vllm-metal 0.31.0, MLX 0.32.1, torch 2.13.0. Edits to Python in either clone apply on restart;
edits to `.metal` shaders or `paged_ops.cpp` need `VLLM_METAL_BUILD_FROM_SOURCE=1` on the next start.

## Commands

```sh
cd ~/dev/spelunking/experiments
python lab.py check                                    # lint the filing
cd ~/dev/vllm-lab/vllm-metal && python tools/check_parity.py --model Qwen/Qwen3-0.6B --top-k 5   # correctness vs mlx_lm
```
