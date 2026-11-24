# Speculative decoding variants — launchers and benchmarks

Server starters for Qwen3.6-35B-A3B-int4-sym-g128-awq (GPU, prefix caching on).
All measured with the sampled-decoding relaxes from our `sampled-linear-verifier`
GenAI build (prompt lookup / scheduler / MTP / DFlash admission: greedy || multinomial).

## Server starters

| Variant | Starter | do_sample | Speed | Notes |
|---|---|---|---|---|
| Prompt lookup + sampled | `Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-LAPLD-sampled.bat` | true | **110–115 tok/s** on edit profile (acceptance 83–85 %), 33–38 on prose | **Main starter for agent sessions** — best on edit/echo-heavy tasks |
| Prompt lookup + greedy | `Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-LAPLD-dev.bat` | false | 34–37 tok/s | The pre-relax default |
| DFlash2 + sampled | `Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-dflash2-sampled.bat` | true | 55 tok/s edit, **58 tok/s** prose (acceptance ~50 %) | Best on free-form generation; needs `max_ngram_size: 0` |
| DFlash2 + greedy | `Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-dflash2-dev.bat` | false | 50–58 tok/s | Draft: `models/Qwen3.6-35B-A3B-dflash2-int4-sym-g128` |
| MTP + sampled | `Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-MTP-sampled.bat` | true | ~36–38 tok/s prose, 34–36 edit (acceptance ~38 %) | Needs free VRAM: two pipelines + LA checkpoints; an unrelated process holding GPU memory will cause OOM |
| No draft + sampled | `Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-nodraft-sampled.bat` | true | 39–41 tok/s | Safe baseline, nothing to mispredict |
| No draft + greedy | `Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-nodraft-greedy.bat` | false | ~34 tok/s | Baseline |

## Benchmark scripts (`scripts/`, self-contained: start their own server on port 8897+)

| Script | Measures | Configs (`python script.py <config>`) |
|---|---|---|
| `edit_file_bench.py` | edit/echo profile — output literally continues the prompt; tok/s + acceptance | `edit_pl_sampled`, `edit_nodraft`, `edit_dflash2_sampled`, `edit_dflash2_greedy`, `edit_mtp_sampled` |
| `unittest_bench.py` | paraphrasing profile — pytest tests for `tool_call.py` (acceptance is low here) | `pl_sampled`, `nodraft`, `pl_greedy` |
| `tv_exactness_test.py` | exactness of sampled spec-decoding vs pure sampling: TV between empirical 4-gram distributions (methodology of upstream #4769) | `--mode draft` / `--mode nodraft` / `--mode compare` |
| `sampled_pc_test.py` | smoke: distinct seeds → distinct outputs (prompt lookup) | — |
| `mtp_sampled_pc_test.py` | smoke: MTP + prefix caching + sampling | — |
| `nodraft_baseline_test.py` / `greedy_baseline_test.py` | baseline speed | — |

## Key numbers (edit profile, T=0.8, seed 7, 1200-token budget)

| Config | tok/s | acceptance |
|---|---|---|
| sampled prompt lookup | 110–115 | 83–85 % |
| greedy prompt lookup | 36.7 (unit-test profile) | — |
| DFlash2 sampled | 52.9–55.3 | 48–50 % |
| DFlash2 greedy | 57–58 | — |
| sampled no draft | 39–40 | — |

## Decision rule

Acceptance rate is the deciding factor: prompt lookup wins when the output
literally continues the prompt (agent `edit_file` calls — 83–85 % acceptance,
2.9x over pure sampling); on paraphrasing/free-form generation the learned
DFlash2 draft holds ~50 % acceptance and beats both prompt lookup and the
no-draft baseline.

## GenAI build

The build is `openvino.genai` branch `sampled-linear-verifier` = upstream master
+ 4 relaxes (prompt lookup verifier, scheduler, MTP admission, DFlash admission).
Update + rebuild + deploy: `C:\alex\github\openvinotoolkit\update-genai.bat`
(`full` argument for a clean build after big master jumps). PR: m4gshm/openvino.genai#2.
