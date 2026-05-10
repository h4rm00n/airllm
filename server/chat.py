#!/usr/bin/env python3
"""
Terminal chat with AirLLM Qwen3.5 model.

Usage:
    python server/chat.py --model-path /mnt/d/AI/models/qwen35fp8 --device cuda:0

Type /exit to quit, /clear to reset conversation.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    from airllm import AutoModel

    print(f"Loading model from {args.model_path} ...")
    model = AutoModel.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        max_seq_len=args.max_seq_len,
        prefetching=False,
    )
    tokenizer = model.tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model_name = os.path.basename(args.model_path.rstrip("/"))
    print(f"Model '{model_name}' ready.\n")

    messages: list[dict[str, str]] = []
    system_prompt = "You are a helpful assistant."

    while True:
        try:
            user_input = input("\033[92mYou > \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.strip() == "/exit":
            break
        if user_input.strip() == "/clear":
            messages = []
            print("Conversation cleared.\n")
            continue
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        full_msgs = [{"role": "system", "content": system_prompt}] + messages

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in full_msgs)

        inputs = tokenizer(prompt, return_tensors="pt")
        n_prompt = inputs.input_ids.shape[1]
        print(f"\033[90m(prompt: {n_prompt} tokens)\033[0m")

        print("\033[93mBot > \033[0m", end="", flush=True)

        output_ids = model.generate(
            inputs.input_ids.to(args.device),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0.01,
            temperature=args.temperature if args.temperature > 0.01 else None,
            use_cache=True,
        )

        new_ids = output_ids[0, n_prompt:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(response)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
