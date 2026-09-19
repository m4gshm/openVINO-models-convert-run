"""
Demonstration of MTP (Multi-Token Prediction) speculative decoding with Qwen3.5-2B.

MTP predicts one extra token per step using the main model's last hidden state.
This script shows:
1. Normal autoregressive generation (main model only)
2. MTP-assisted speculative generation where MTP drafts a token and the main model
   verifies it in the next step
3. Statistics on how many consecutive MTP predictions are accepted
"""

import numpy as np
import openvino as ov
from transformers import AutoTokenizer
from pathlib import Path

MODEL_DIR = Path("../models/Qwen3.5-2B-int8-sym")

F32_CONFIG = {}


def load_models(device="GPU"):
    core = ov.Core()

    print("Loading models...")
    text_embed = core.compile_model(MODEL_DIR / "openvino_text_embeddings_model.xml", device, F32_CONFIG)
    print("  Text embeddings model loaded.")

    # Trace back from the logits Result to the final lm_head MatMul.
    #   logits = norm_hidden @ lm_head_weight.T   (transpose_b=True)
    #   input(0) -> post-norm hidden state
    #   input(1) -> lm_head weight [vocab, hidden]  (Convert node over a bf16 constant)
    # We add BOTH as extra outputs: the hidden state (to feed the MTP model) and
    # the true lm_head weight (to decode MTP hidden states into logits).
    #
    # IMPORTANT: the lm_head weight must come from the language model, NOT from the
    # text-embeddings model. They are only identical when tie_word_embeddings=True
    # (e.g. Qwen3.5-2B). Models like Qwen3.6-35B-A3B have tie_word_embeddings=False
    # with a separate lm_head; using the embedding matrix there produces garbage MTP
    # logits and a 0% acceptance rate.
    lm_model = core.read_model(MODEL_DIR / "openvino_language_model.xml")
    logits_result = lm_model.get_results()[0]
    matmul_node = logits_result.input(0).get_source_output().get_node()
    hidden_node = matmul_node.input(0).get_source_output()
    lm_head_node = matmul_node.input(1).get_source_output()
    # Ensure the lm_head weight is emitted as f32 (the source is a bf16 constant,
    # typically already followed by a Convert; add one defensively if needed).
    if lm_head_node.get_element_type() != ov.Type.f32:
        lm_head_node = ov.opset13.convert(lm_head_node, ov.Type.f32).output(0)
    lm_model.add_results([ov.opset13.result(hidden_node), ov.opset13.result(lm_head_node)])
    language_model = core.compile_model(lm_model, device, F32_CONFIG)
    del lm_model
    print("  Language model loaded (with last_hidden_state + lm_head outputs).")

    mtp_model = core.compile_model(MODEL_DIR / "openvino_mtp_model.xml", device, F32_CONFIG)
    print("  MTP model loaded.")

    # Materialize the lm_head weight by running the language model once with a
    # trivial input. The weight is a constant, so any single inference exposes it.
    lm_head_output_idx = len(language_model.outputs) - 1
    warmup_req = language_model.create_infer_request()
    seq = 1
    warmup_req.infer({
        "inputs_embeds": np.zeros((1, seq, text_embed.output(0).get_partial_shape()[2].get_length()), dtype=np.float32),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
        "position_ids": np.zeros((4, 1, seq), dtype=np.int64),
        "beam_idx": np.array([0], dtype=np.int32),
    })
    lm_head_weight = warmup_req.get_output_tensor(lm_head_output_idx).data.copy()
    del warmup_req
    print(f"  LM head weight: {lm_head_weight.shape}, range=[{lm_head_weight.min():.4f}, {lm_head_weight.max():.4f}]")

    return text_embed, language_model, mtp_model, lm_head_weight


def get_embeddings(text_embed, token_ids):
    """Get embeddings for token IDs using the text embeddings model."""
    result = text_embed({"input": token_ids})
    return result["inputs_embeds"].astype(np.float32)


class LMState:
    """Manages the language model inference state."""

    def __init__(self, language_model, text_embed):
        self.lm = language_model
        self.text_embed = text_embed
        self.lm_request = self.lm.create_infer_request()
        self.position = 0
        # Two extra outputs were appended via add_results, in this order:
        #   [..., last_hidden_state, lm_head_weight]
        # so the hidden state is the second-to-last output.
        self._hidden_output_idx = len(self.lm.outputs) - 2

    def reset(self):
        self.lm_request.reset_state()
        self.position = 0

    def prefill(self, token_ids):
        """Process prompt tokens and return logits + hidden state for all positions."""
        embeds = get_embeddings(self.text_embed, token_ids)
        seq_len = token_ids.shape[1]

        # MRoPE position_ids: [4, batch, seq] — for text-only, all dims are the same
        position_ids = np.arange(self.position, self.position + seq_len, dtype=np.int64)
        position_ids = position_ids.reshape(1, 1, -1).repeat(4, axis=0)  # [4, 1, seq]

        attention_mask = np.ones((1, self.position + seq_len), dtype=np.int64)

        self.lm_request.infer({
            "inputs_embeds": embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "beam_idx": np.array([0], dtype=np.int32),
        })

        self.position += seq_len
        logits = self.lm_request.get_tensor("logits").data.copy()
        hidden = self.lm_request.get_output_tensor(self._hidden_output_idx).data.copy()
        return logits, hidden

    def decode_one(self, token_id):
        """Decode a single token and return logits + hidden state."""
        token_ids = np.array([[token_id]], dtype=np.int64)
        embeds = get_embeddings(self.text_embed, token_ids)

        position_ids = np.array([self.position], dtype=np.int64).reshape(1, 1, 1).repeat(4, axis=0)
        attention_mask = np.ones((1, self.position + 1), dtype=np.int64)

        self.lm_request.infer({
            "inputs_embeds": embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "beam_idx": np.array([0], dtype=np.int32),
        })

        self.position += 1
        logits = self.lm_request.get_tensor("logits").data.copy()
        hidden = self.lm_request.get_output_tensor(self._hidden_output_idx).data.copy()
        return logits, hidden


class MTPState:
    """Manages the MTP model inference state."""

    def __init__(self, mtp_model, text_embed, lm_head_weight):
        self.mtp = mtp_model
        self.text_embed = text_embed
        # True lm_head weight [vocab, hidden] taken from the language model IR.
        # Used to project MTP hidden states into vocabulary logits.
        self.lm_head_weight = lm_head_weight
        self.mtp_request = self.mtp.create_infer_request()
        self.position = 0

    def reset(self):
        self.mtp_request.reset_state()
        self.position = 0

    def prefill(self, hidden_states_all, token_ids, first_generated_token):
        """
        Populate MTP KV cache by processing all prompt positions token-by-token.

        MTP at position i receives:
          hidden_states = main_model.last_hidden_state[i]
          inputs_embeds = embed(token[i+1])  (the NEXT token's embedding)
        This mirrors the training objective: predict token[i+2] given hidden[i] and embed(token[i+1]).

        Args:
            hidden_states_all: [1, N, hidden_size] - main model hidden states for all prompt positions
            token_ids: [1, N] - prompt token IDs
            first_generated_token: int - the first token predicted by main model (argmax of logits[-1])
        """
        seq_len = token_ids.shape[1]

        # Build all MTP inputs in one batch:
        # positions 0..N-2: hidden[i] + embed(token[i+1])
        # position N-1: hidden[N-1] + embed(first_generated_token)
        all_hidden = hidden_states_all.astype(np.float32)  # [1, N, hidden_size]

        # Next-token IDs: token[1], token[2], ..., token[N-1], first_generated_token
        next_token_ids = np.concatenate([
            token_ids[:, 1:],  # [1, N-1]
            np.array([[first_generated_token]], dtype=np.int64),
        ], axis=1)  # [1, N]
        all_embeds = get_embeddings(self.text_embed, next_token_ids)  # [1, N, hidden_size]

        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        attention_mask = np.ones((1, seq_len), dtype=np.int64)

        self.mtp_request.infer({
            "hidden_states": all_hidden,
            "inputs_embeds": all_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "beam_idx": np.array([0], dtype=np.int32),
        })
        self.position = seq_len

        # Return prediction (MTP predicts token after first_generated_token)
        mtp_hidden = self.mtp_request.get_tensor("last_hidden_state").data.copy().astype(np.float32)
        logits = mtp_hidden @ self.lm_head_weight.T
        return int(np.argmax(logits[0, -1, :]))

    def predict(self, hidden_states, token_id):
        """
        Given the main model's hidden state and the predicted token,
        predict the next token via MTP.

        Args:
            hidden_states: [1, 1, hidden_size] from main model's last_hidden_state
            token_id: the token that was just predicted by main model
        Returns:
            predicted_token_id (greedy argmax of MTP logits)
        """
        token_ids = np.array([[token_id]], dtype=np.int64)
        embeds = get_embeddings(self.text_embed, token_ids)

        # Use only the last position's hidden state
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]
        hidden_states = hidden_states.astype(np.float32)

        position_ids = np.array([[self.position]], dtype=np.int64)
        attention_mask = np.ones((1, self.position + 1), dtype=np.int64)

        self.mtp_request.infer({
            "hidden_states": hidden_states,
            "inputs_embeds": embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "beam_idx": np.array([0], dtype=np.int32),
        })

        self.position += 1

        # MTP outputs last_hidden_state; apply lm_head to get logits
        mtp_hidden = self.mtp_request.get_tensor("last_hidden_state").data.copy().astype(np.float32)
        logits = mtp_hidden @ self.lm_head_weight.T
        predicted_token = int(np.argmax(logits[0, -1, :]))
        return predicted_token


def generate_baseline(lm_state, tokenizer, prompt, max_tokens=50):
    """Standard autoregressive generation (no MTP)."""
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]

    lm_state.reset()
    logits, _ = lm_state.prefill(input_ids)

    generated = []
    next_token = int(np.argmax(logits[0, -1, :]))
    generated.append(next_token)

    for _ in range(max_tokens - 1):
        if next_token == tokenizer.eos_token_id:
            break
        logits, _ = lm_state.decode_one(next_token)
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)

    return generated


def generate_with_mtp(lm_state, mtp_state, tokenizer, prompt, max_tokens=50):
    """
    Speculative generation using MTP.

    MTP semantics:
      MTP(hidden[t], embed(token_predicted_at_t)) → predicts token at t+1
      Where hidden[t] is the main model's last_hidden_state at step t,
      and token_predicted_at_t is what main model predicts at step t.

    Flow:
    1. Prefill: main model processes prompt → hidden[0..N-1], logits
    2. First generated token = argmax(logits[-1])
    3. MTP prefill: feed (hidden[i], embed(token[i+1])) for i=0..N-2 to build KV,
       then (hidden[N-1], embed(first_gen_token)) → draft for position N+1
    4. Decode loop:
       - Main model decodes token → new hidden + logits → new main_token
       - Verify: does draft == main_token?
       - MTP(new_hidden, embed(main_token)) → next draft
    """
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]

    lm_state.reset()
    mtp_state.reset()

    # Main model prefill — get hidden states for ALL positions
    logits, hidden = lm_state.prefill(input_ids)

    # First generated token from main model
    next_token = int(np.argmax(logits[0, -1, :]))

    # MTP prefill: process all positions, last step predicts token AFTER next_token
    draft_token = mtp_state.prefill(hidden, input_ids, next_token)

    generated = [next_token]
    accepted = 0
    rejected = 0
    consecutive_accepted = []
    current_streak = 0

    for _ in range(max_tokens - 1):
        if next_token == tokenizer.eos_token_id:
            break

        # Main model decodes next_token → logits for the NEXT position
        logits, hidden = lm_state.decode_one(next_token)
        main_token = int(np.argmax(logits[0, -1, :]))

        # Check if MTP's draft matches main model's greedy choice
        if draft_token == main_token:
            accepted += 1
            current_streak += 1
        else:
            rejected += 1
            if current_streak > 0:
                consecutive_accepted.append(current_streak)
            current_streak = 0

        # Use main model's token (ground truth)
        next_token = main_token
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        # MTP predicts next draft: hidden from current step + embed(main_token)
        draft_token = mtp_state.predict(hidden, next_token)

    if current_streak > 0:
        consecutive_accepted.append(current_streak)

    return generated, accepted, rejected, consecutive_accepted


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    text_embed, language_model, mtp_model, lm_head_weight = load_models()

    lm_state = LMState(language_model, text_embed)
    mtp_state = MTPState(mtp_model, text_embed, lm_head_weight)

    prompts = [
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,",
        "Once upon a time there was a little girl named Goldilocks. She went for a walk in the forest. Pretty soon, she came upon a house. She knocked and",
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the",
    ]

    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: {repr(prompt)}")
        print(f"{'='*70}")

        # Baseline generation
        baseline_tokens = generate_baseline(lm_state, tokenizer, prompt, max_tokens=30)
        baseline_text = tokenizer.decode(baseline_tokens, skip_special_tokens=True)
        print(f"\nBaseline output: {baseline_text}")

        # MTP-assisted generation
        generated, accepted, rejected, streaks = generate_with_mtp(
            lm_state, mtp_state, tokenizer, prompt, max_tokens=30
        )
        mtp_text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"MTP output:      {mtp_text}")

        # Verify outputs match (they should with greedy decoding)
        match = baseline_tokens == generated
        print(f"\nOutputs match: {match}")

        total = accepted + rejected
        if total > 0:
            print(f"\nMTP Statistics:")
            print(f"  Accepted: {accepted}/{total} ({100*accepted/total:.1f}%)")
            print(f"  Rejected: {rejected}/{total} ({100*rejected/total:.1f}%)")
            if streaks:
                print(f"  Consecutive acceptance streaks: {streaks}")
                print(f"  Max consecutive accepted: {max(streaks)}")
                print(f"  Avg consecutive accepted: {sum(streaks)/len(streaks):.1f}")


if __name__ == "__main__":
    main()