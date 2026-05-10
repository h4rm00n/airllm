#!/usr/bin/env python3
"""
OpenAI-compatible API server for AirLLM models.

Usage:
    python server/openai_server.py \
        --model-path /path/to/model \
        --device cuda:0 \
        --dtype float16 \
        --max-seq-len 4096 \
        --port 8000 \
        --host 0.0.0.0

Endpoints:
    POST /v1/chat/completions   — Chat completion (OpenAI-compatible)
    POST /v1/completions        — Text completion (OpenAI-compatible)
    GET  /v1/models             — List models
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
import threading
from typing import Optional, Generator

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("airllm-server")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── OpenAI-compatible request/response schemas ──────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    max_tokens: int = Field(default=256, alias="max_tokens")
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[list[str]] = None
    enable_thinking: bool = False

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    max_tokens: int = Field(default=256, alias="max_tokens")
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[list[str]] = None

class Choice(BaseModel):
    index: int
    message: Optional[dict] = None
    text: Optional[str] = None
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "airllm"

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ── Generation wrapper (sequential access) ──────────────────────────────

class AirLLMGenerator:
    _THINK_PATTERN = re.compile(r"<think\s*>(.*?)</think\s*>", re.DOTALL)

    def __init__(self, model_path: str, device: str = "cuda:0",
                 dtype: str = "float16", max_seq_len: int = 4096):
        from airllm import AutoModel
        self.model_path = model_path
        self.device = device
        self._dtype = getattr(torch, dtype)
        self.max_seq_len = max_seq_len
        self.model = AutoModel.from_pretrained(
            model_path, device=device, dtype=self._dtype,
            max_seq_len=max_seq_len, prefetching=False,
        )
        self.tokenizer = self.model.tokenizer
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return os.path.basename(self.model_path.rstrip("/"))

    def _prepare_kwargs(self, max_tokens: int, temperature: float, top_p: float,
                        stop: list[str] | None = None) -> dict:
        kwargs = {"max_new_tokens": max_tokens, "use_cache": True}
        if temperature < 0.01:
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        if stop:
            stop_ids = []
            for s in stop:
                ids = self.tokenizer(s, add_special_tokens=False).input_ids
                stop_ids.extend(ids)
            if stop_ids:
                kwargs["eos_token_id"] = list(set(
                    [self.tokenizer.eos_token_id] + stop_ids
                ))
        return kwargs

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 1.0,
                 stop: list[str] | None = None) -> tuple[str, int, int]:
        with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            prompt_len = input_ids.shape[1]
            gen_kwargs = self._prepare_kwargs(max_tokens, temperature, top_p, stop)
            output_ids = self.model.generate(input_ids, **gen_kwargs)
            new_ids = output_ids[0, prompt_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            return text, prompt_len, len(new_ids)

    def generate_stream(self, prompt: str, max_tokens: int = 256,
                        temperature: float = 0.7, top_p: float = 1.0,
                        stop: list[str] | None = None) -> Generator[str, None, None]:
        """True streaming: yields text chunks as they are generated."""
        with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            prompt_len = input_ids.shape[1]

            eos_ids = {self.tokenizer.eos_token_id}
            if stop:
                for s in stop:
                    ids = self.tokenizer(s, add_special_tokens=False).input_ids
                    for tid in ids:
                        eos_ids.add(tid)

            logger.info(f"Stream generate: prompt_len={prompt_len}, max_new={max_tokens}")

            past_kv = None
            generated_ids = []
            prev_text = ""

            outputs = self.model(input_ids=input_ids, use_cache=True)
            past_kv = outputs.past_key_values

            for step in range(max_tokens):
                logits = outputs.logits[:, -1, :]
                if temperature < 0.01:
                    next_id = logits.argmax(dim=-1)
                else:
                    logits = logits / max(temperature, 0.01)
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = logits.sort(descending=True)
                        cumsum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                        cutoff = (cumsum > top_p).int().argmax(dim=-1) + 1
                        logits[:, sorted_indices[0, cutoff:]] = float('-inf')
                    probs = logits.softmax(dim=-1)
                    next_id = probs.multinomial(1)

                token_id = next_id.item()
                generated_ids.append(token_id)
                full_text = self.tokenizer.decode(generated_ids,
                                                  skip_special_tokens=True)
                new_text = full_text[len(prev_text):]
                if new_text:
                    yield new_text
                prev_text = full_text

                if token_id in eos_ids:
                    break

                outputs = self.model(
                    input_ids=next_id.view(1, 1),
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = outputs.past_key_values

            logger.info(f"Stream done: {len(generated_ids)} tokens generated")

    @staticmethod
    def _parse_thinking_content(text: str) -> tuple[str, Optional[str]]:
        match = AirLLMGenerator._THINK_PATTERN.search(text)
        if match:
            reasoning = match.group(1).strip()
            content = text[match.end():].strip()
            return content, reasoning
        return text.strip(), None

    def chat_generate(self, messages: list[dict], max_tokens: int = 256,
                      temperature: float = 0.7, top_p: float = 1.0,
                      stop: list[str] | None = None,
                      enable_thinking: bool = False) -> tuple[str, int, int, Optional[str]]:
        prompt = self._build_chat_prompt(messages, enable_thinking=enable_thinking)
        text, prompt_tokens, completion_tokens = self.generate(
            prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, stop=stop,
        )
        if enable_thinking:
            content, reasoning = self._parse_thinking_content(text)
            return content, prompt_tokens, completion_tokens, reasoning
        return text, prompt_tokens, completion_tokens, None

    def chat_generate_stream(self, messages: list[dict], max_tokens: int = 256,
                             temperature: float = 0.7, top_p: float = 1.0,
                             stop: list[str] | None = None,
                             enable_thinking: bool = False) -> Generator[tuple[str, Optional[str]], None, None]:
        prompt = self._build_chat_prompt(messages, enable_thinking=enable_thinking)
        thinking_buffer = ""
        think_open = False
        think_closed = False
        for chunk in self.generate_stream(prompt, max_tokens, temperature, top_p, stop):
            if not enable_thinking:
                yield chunk, None
                continue
            thinking_buffer += chunk
            if not think_closed:
                if not think_open:
                    idx = thinking_buffer.find("<think")
                    if idx != -1:
                        before = thinking_buffer[:idx]
                        if before.strip():
                            yield before, None
                        thinking_buffer = thinking_buffer[idx:]
                        think_open = True
                if think_open:
                    close_idx = thinking_buffer.find("</think")
                    if close_idx != -1:
                        end_idx = thinking_buffer.find(">", close_idx)
                        if end_idx != -1:
                            reasoning = thinking_buffer[len("<think"):close_idx]
                            if reasoning.startswith(">"):
                                reasoning = reasoning[1:]
                            thinking_buffer = thinking_buffer[end_idx + 1:]
                            yield thinking_buffer, reasoning.strip()
                            thinking_buffer = ""
                            think_closed = True
                    else:
                        reasoning_so_far = thinking_buffer[len("<think"):]
                        if reasoning_so_far.startswith(">"):
                            reasoning_so_far = reasoning_so_far[1:]
                        yield "", reasoning_so_far
                        thinking_buffer = ""
            else:
                if chunk:
                    yield chunk, None

    def _build_chat_prompt(self, messages: list[dict],
                           enable_thinking: bool = False) -> str:
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


# ── FastAPI app ─────────────────────────────────────────────────────────

def create_app(generator: AirLLMGenerator) -> FastAPI:
    app = FastAPI(title="AirLLM OpenAI API", version="1.0.0")

    @app.get("/v1/models")
    async def list_models():
        return ModelList(data=[ModelInfo(
            id=generator.model_name,
            created=int(time.time()),
        )])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        messages = [{"role": m.role, "content": m.content}
                    for m in req.messages]
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if req.stream:
            async def event_stream():
                gen = generator.chat_generate_stream(
                    messages, max_tokens=req.max_tokens,
                    temperature=req.temperature, top_p=req.top_p,
                    stop=req.stop, enable_thinking=req.enable_thinking,
                )
                loop = asyncio.get_event_loop()
                _sentinel = object()

                def _next_gen(g):
                    v = next(g, _sentinel)
                    if v is _sentinel:
                        raise StopAsyncIteration
                    return v

                try:
                    while True:
                        try:
                            tok = await loop.run_in_executor(None, _next_gen, gen)
                        except StopAsyncIteration:
                            break
                        content, reasoning = tok
                        delta: dict = {}
                        if reasoning:
                            delta["reasoning_content"] = reasoning
                        if content:
                            delta["content"] = content
                        if delta:
                            yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': generator.model_name, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"
                except Exception as e:
                    import traceback
                    logger.error(f"Stream error: {e}\n{traceback.format_exc()}")
                finally:
                    try:
                        yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': generator.model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    except Exception:
                        pass
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                         "X-Accel-Buffering": "no"},
            )

        text, prompt_tokens, completion_tokens, reasoning = generator.chat_generate(
            messages, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p, stop=req.stop,
            enable_thinking=req.enable_thinking,
        )
        msg: dict = {"role": "assistant", "content": text}
        if reasoning is not None:
            msg["reasoning_content"] = reasoning
        return ChatCompletionResponse(
            id=req_id, created=int(time.time()), model=generator.model_name,
            choices=[Choice(index=0, message=msg, finish_reason="stop")],
            usage=Usage(prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens),
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"

        if req.stream:
            async def event_stream():
                gen = generator.generate_stream(
                    req.prompt, max_tokens=req.max_tokens,
                    temperature=req.temperature, top_p=req.top_p,
                    stop=req.stop,
                )
                loop = asyncio.get_event_loop()
                _sentinel = object()

                def _next_gen(g):
                    v = next(g, _sentinel)
                    if v is _sentinel:
                        raise StopAsyncIteration
                    return v

                try:
                    while True:
                        try:
                            tok = await loop.run_in_executor(None, _next_gen, gen)
                        except StopAsyncIteration:
                            break
                        yield f"data: {json.dumps({'id': req_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': generator.model_name, 'choices': [{'index': 0, 'text': tok, 'finish_reason': None}]})}\n\n"
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                else:
                    yield f"data: {json.dumps({'id': req_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': generator.model_name, 'choices': [{'index': 0, 'text': '', 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                         "X-Accel-Buffering": "no"},
            )

        text, pt, ct = generator.generate(
            req.prompt, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p, stop=req.stop,
        )
        return CompletionResponse(
            id=req_id, created=int(time.time()), model=generator.model_name,
            choices=[Choice(index=0, text=text, finish_reason="stop")],
            usage=Usage(prompt_tokens=pt, completion_tokens=ct,
                        total_tokens=pt + ct),
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": generator.model_name}

    return app


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AirLLM OpenAI Server")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    generator = AirLLMGenerator(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
    )
    print(f"Model '{generator.model_name}' loaded. Starting server...")

    app = create_app(generator)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
