# Speculative Decoding Draft Models for Qwen3.6-35B-A3B

Target: `Qwen/Qwen3.6-35B-A3B` — vocab 248320, hidden 2048, hybrid linear attention
(local export: `models/Qwen3.6-35B-A3B-int4-sym-g128-awq`, arch `Qwen3_5MoeForConditionalGeneration`).

## Greedy constraint

The scheduler runs with `enable_prefix_caching: true` (required for speed) and the model has a
linear-attention cache. openvino.genai asserts in
`continuous_batching/pipeline_impl.cpp::_validate_linear_verifier_constraints`:

> Prefix linear-attention verification requires greedy decoding and a supported strategy for embedding input

So **every speculative strategy on this model runs greedy only** while prefix caching is on
(the same assert exists in `mtp_strategy.cpp`). Sampling knobs (`temperature`, `top_p`, `top_k`,
`min_p`) are inert under `do_sample=false`; the active anti-repetition knobs are the penalties
(`repetition_penalty`, `presence_penalty`, `frequency_penalty`) — `LogitProcessor` applies them
for both multinomial and greedy decoding, before the argmax.

Agent-side consequence: loop retries that raise the temperature only work when the request
actually samples. Under greedy the agent skips retries and asks the user (continue/interrupt)
right after the first detected loop.

## Draft candidates

| Strategy | Sampling | Draft candidate | Vocab / hidden | Status |
| --- | --- | --- | --- | --- |
| Fast Draft (classic SD) | greedy+multinomial | `models/Qwen3.5-2B-int4-ov` (already converted, `Qwen3_5ForCausalLM`, 24 layers) | 248320 / 2048 — matches | ready, zero conversion |
| EAGLE3 (chain) | greedy+multinomial | `Dogacel/specdrift-qwen3.6-35b-a3b-eagle3` (`LlamaForCausalLMEagle3`, 1 layer, carries `d2t`) | 248320 / 2048 — matches | converted: `models/Qwen3.6-35B-A3B-eagle3-int4-sym-g128` via `src_models/init-Dogacel-specdrift-qwen3.6-35b-a3b-eagle3-int4-sym-g128.bat` |
| EAGLE3 (chain) | greedy+multinomial | `circulus/Qwen3.6-35B-A3B-eagle3-ov-int4` (already OV) | 248320 / 2048 | community export; rt_info / d2t not verified — prefer re-export from Dogacel |
| EAGLE3 (tree) | greedy only | same heads as above | — | tree search needs greedy anyway |
| DFlash v2 (WORKING) | greedy only | `incoai/Qwen3.6-35B-A3B-DFlash2` (`DFlash2DraftModel`, 6 layers) | 248320 / 2048 | **working: 50–87% acceptance, ~2.6x** — `models/Qwen3.6-35B-A3B-dflash2-int4-sym-g128` via `src_models/init-incoai-DFlash2-Qwen3.6-35B-A3B-int4-sym-g128.bat`; needs optimum-intel git-master AND a GenAI build with gated-head support (see below) |
| DFlash v1 | greedy only | `AEON-7/AEON-DFlash-Qwen3.6-35B-A3B` (`DFlashDraftModel`, 8 layers), `z-lab/Qwen3.6-35B-A3B-DFlash` | 248320 / 2048 | same |
| DSpark | greedy only | — | — | skipped: greedy only, no Qwen3.6 head found |
| MTP (current dev bat) | greedy only (prefix caching) | target's own `openvino_mtp_model.xml` | — | running |

EAGLE3/DFlash heads are trained against the target's hidden states — a draft trained for another
model family does not fit even with matching vocab/hidden.

GenAI reads the EAGLE3 aux layers from the draft IR's rt_info (`apply_eagle3_rt_info` in
`llm/pipeline.cpp`): without `hidden_layers_list` it defaults to `{2, n/2, n-3}` computed from the
target config. Community drafts can be trained on different layers (Dogacel specdrift: `[3, 19, 35]`)
— with the default the graft finds no residual nodes on linear-attention layers and fails with
"Port for tensor name last_hidden_state was not found". The post-export step
`src_models/apply_eagle3_rt_info.py <draft_dir>` copies
`eagle_config.eagle_aux_hidden_state_layer_ids` from the draft config.json into the IR rt_info
(also required: `eagle3_mode=true`). Only full-attention layers can serve as aux layers — in this
target those are layers 3, 7, ..., 39.

When the main model is a VLM (`--pipe VLM`), the eagle3 strategy always runs the draft in
EMBEDDINGS mode: the runner sets `inputs_embeds` on the draft by name. A tokens-style draft
(`input_ids` → embed_tokens `Gather`) has no such port and the pipeline start fails with
"Port inputs_embeds was not found". The post-export step
`src_models/make_eagle3_draft_embeds_native.py <draft_dir>` converts the draft to an
embeds-native one: adds the `inputs_embeds` Parameter (f32 `[-1,-1,H]`), re-points the embed
Gather consumers to it, re-points the `ShapeOf(input_ids)` KV-cache shape chain to
`attention_mask` (same `[b, s]` rank) and drops the dead `input_ids`/Gather subgraph
(also shrinks the bin by the now-unused vocab embedding table). `hidden_states` stays
dynamic — in the file it feeds the 6144-wide fc (required by GenAI's
`move_fc_from_draft_to_main`), at runtime GenAI feeds the 2048-wide fc output instead.

## Conversion commands

```bat
@REM EAGLE3 draft head (chain drafting):
optimum-cli export openvino ^
  --model Dogacel/specdrift-qwen3.6-35b-a3b-eagle3 ^
  --task text-generation-with-past ^
  --weight-format int4 ^
  --backup-precision int8_sym ^
  --group-size 128 ^
  --trust-remote-code ^
  --sym ^
  ../models/Qwen3.6-35B-A3B-eagle3-int4-sym-g128
@REM required post-steps (rt_info: eagle3_mode + hidden_layers_list; embeds-native draft for VLM mains):
python src_models/apply_eagle3_rt_info.py ../models/Qwen3.6-35B-A3B-eagle3-int4-sym-g128
python src_models/make_eagle3_draft_embeds_native.py ../models/Qwen3.6-35B-A3B-eagle3-int4-sym-g128

@REM DFlash draft (same pattern as src_models/init-Qwen3.6-35B-A3B-DFlash-int4-sym-g128.bat):
@REM WORKING variant — see "DFlash2 conversion walk-through" below; run
@REM src_models/init-incoai-DFlash2-Qwen3.6-35B-A3B-int4-sym-g128.bat instead of hand-typing.
optimum-cli export openvino ^
  --model incoai/Qwen3.6-35B-A3B-DFlash2 ^
  --task text-generation-with-past ^
  --weight-format int4 ^
  --backup-precision int8_sym ^
  --group-size 128 ^
  --trust-remote-code ^
  --sym ^
  ../models/Qwen3.6-35B-A3B-DFlash2-int4-sym-g128
```

## DFlash2 conversion walk-through (incoai, gated head — the working draft)

Source: `incoai/Qwen3.6-35B-A3B-DFlash2` (1.05 GB safetensors, no tokenizer in the repo).
Draft: 6 sliding-attention layers, `is_causal: false` (bidirectional over a mask-token block),
block_size 8 (seed + 7 drafted tokens per step), reads the target's hidden states at layers
`[1, 6, 11, 16, 22, 27, 32, 37]`, **full target vocab 248320 — no d2t mapping**, and a separate
**gated Markov head** (`markov_head_type: dflash2`, rank 256, top_k 16) exported by optimum-intel
as `openvino_markov_head_model.xml` next to the draft backbone. Unlike the DSpark "vanilla" head,
the gated candidate scorer reads a 4th input, `draft_hidden_states` — the backbone hidden states
of the scored rows.

Steps (all of it is `src_models/init-incoai-DFlash2-Qwen3.6-35B-A3B-int4-sym-g128.bat`):

1. **Toolchain.** `pip install optimum-intel@git+https://github.com/huggingface/optimum-intel.git`
   — release 2.3.0 has no `DFlash2DraftModel` patcher and no gated-head export. Verify with
   `optimum/exporters/openvino/model_patcher.py` containing `"DFlash2DraftModel": "Qwen3DFlash2ForCausalLM"`.
2. **Download** `snapshot_download('incoai/Qwen3.6-35B-A3B-DFlash2', local_dir=...)`. Note
   `python -m huggingface_hub.commands.huggingface_cli` is broken in this python (module layout
   changed) — use the API.
3. **Borrow the tokenizer** from the target (`models/Qwen3.6-35B-A3B-int4-sym-g128-awq`):
   copy `tokenizer.json` + `tokenizer_config.json` into the draft dir; the repo ships none and
   neither the exporter nor the GenAI draft loader can live without it.
4. **Export from inside the model dir with `--model .`**. Passing the folder path (or HF id)
   from outside makes optimum-cli re-resolve it as a repo — and the repo id fails with
   "Unrecognized model ... Should have a model_type key" because the draft config carries no
   `model_type` transformers knows. The output path is relative to that cwd too: from
   `src_models/incoai/<model>` it must be `..\..\..\models\...`, otherwise the export lands in
   `src_models\models` (move it by hand if it does).
5. **No post-processing** — the exporter stamps the dflash rt_info block onto the graph itself
   (`dflash_mode=True`, `dflash/mask_token_id`, `dflash/target_layer_ids`, `dflash/markov_rank`,
   `dflash/markov_head_type`). No `apply_eagle3_rt_info.py`, no embeds-native surgery: the
   GenAI dflash transforms graft the target embedding and lm_head into the draft at load time.
   Sanity check: `findstr dflash_mode models\...\openvino_model.xml` must hit, and
   `openvino_markov_head_model.xml` must exist next to the backbone.
6. **GenAI side (gated-head support, patched locally — not in upstream master).** Upstream
   dflash supports only the vanilla DSpark head and builds the candidate scorer with 3 inputs;
   loading a gated draft dies with "Model references undeclared parameters: draft_hidden_states".
   The local patch: `dflash_model_transforms.cpp` accepts `markov_head_type: dflash2`, keeps the
   4-input scorer, and `expose_draft_hidden_rows()` adds a `draft_hidden_states` output to the
   draft graph (the gather-before-matmul rows that feed the grafted lm_head — the same rows as
   the gathered logits, one to one); `dflash_candidate_selector.cpp` feeds it to the scorer
   (flattened `[N,1,H] → [N,H]`, packed like the logits); `pipeline_impl.hpp` wires a getter to
   the live draft infer request. Rebuild `py_openvino_genai` and deploy into `bin\dist`
   (dll → `runtime\bin\intel64\Release`, pyd → `python\openvino_genai`), then restart the server.

Run: `Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-dflash2-dev.bat` (`--dflash_mode on`). Measured:
acceptance 50–87% by prompt, 48 tokens in 1.77 s (~27 t/s, ≈2.6x over the plain baseline;
the eagle3 heads never got off 0%).

Fast Draft needs no conversion: point `--draft_model` at any compatible small export
(`models/Qwen3.5-2B-int4-ov` is ready).

## Run commands

```bat
@REM Fast Draft:
call agent-dev.bat --model Qwen3.6-35B-A3B-int4-sym-g128-awq --draft_model Qwen3.5-2B-int4-ov --detect_cycled_tool_call on --pipe VLM --kv_cache_precision u4 --scheduler_config_file .config/scheduler_config.json

@REM EAGLE3:
call agent-dev.bat --model Qwen3.6-35B-A3B-int4-sym-g128-awq --draft_model Qwen3.6-35B-A3B-eagle3-int4-sym-g128 --eagle3_mode on --detect_cycled_tool_call on --pipe VLM --kv_cache_precision u4

@REM DFlash (WORKING — gated head; GenAI needs the local gated-head patch):
@REM Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-dflash2-dev.bat runs:
call agent-dev.bat --model Qwen3.6-35B-A3B-int4-sym-g128-awq --draft_model Qwen3.6-35B-A3B-dflash2-int4-sym-g128 --dflash_mode on --detect_cycled_tool_call on --pipe VLM --kv_cache_precision u4 --scheduler_config_file .config/scheduler_config.json
```

Measure draft quality by acceptance rate (`extended_perf_metrics.get_num_accepted_tokens()`),
not by wall time alone.

## LAPLD (prompt lookup) and looping

`Qwen3.6-35B-A3B-int4-sym-g128-awq-GPU-LAPLD-dev.bat` runs Prompt Lookup Decoding
(`--prompt_lookup on`, n-gram drafter, no draft model) with
`.config/generate_config_prompt_lookup.json` (`num_assistant_tokens: 5`, `max_ngram_size: 3`,
`do_sample: false`). PLD is subject to the same greedy constraint above.

What loops and what does not:

- `num_assistant_tokens` / `max_ngram_size` change draft length and speed, **not** loop
  occurrence: greedy verification preserves the target distribution. If anything, PLD makes
  loops *faster* — a repeating pattern is exactly what the n-gram drafter proposes, so accepted
  blocks replay the loop in bigger chunks.
- The active anti-loop levers in greedy are the penalties in the generate config:
  - `repetition_penalty` (1.1 in the config): divides positive logits of every token seen in
    prompt or generation. 1.1–1.15 is the working range; it also penalizes legitimate repeats
    (keywords, identifiers) — for coding keep it moderate.
  - `presence_penalty` (0.2 in the config): subtracts a flat value from tokens already
    generated. 0.2–0.5 deters early-loop lock-in with less damage to code text than a high
    repetition penalty.
  - `frequency_penalty` (left at 0.0): subtracts `penalty × count`, grows the longer a loop
    runs — the next escalation step (0.1–0.3) if loops persist; hardest on legitimate code
    repeats.
- `no_repeat_ngram_size` exists in openvino.genai but is applied in beam search only — not a
  knob for greedy.
- Regardless of decoding, the agent loop detectors (`--detect_cycled_tool_call on`) cut the
  attempt and, under greedy, ask the user continue/interrupt immediately.
