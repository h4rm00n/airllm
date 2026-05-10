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
import json
import os
import sys
import time
import uuid
import threading
from queue import Queue
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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
        self._stop_cache: dict[str, list[int]] = {}

    @property
    def model_name(self) -> str:
        return os.path.basename(self.model_path.rstrip("/"))

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 1.0,
                 stop: list[str] | None = None,
                 stream_callback=None) -> tuple[str, int, int]:
        with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            prompt_len = input_ids.shape[1]

            gen_kwargs = {"max_new_tokens": max_tokens, "use_cache": True}
            if temperature < 0.01:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            if stop:
                self._stop_cache[id(self)] = \
                    self.tokenizer(stop, add_special_tokens=False).input_ids

            output_ids = self.model.generate(input_ids, **gen_kwargs)
            new_ids = output_ids[0, prompt_len:]

            if stream_callback:
                for i in range(len(new_ids)):
                    tok = self.tokenizer.decode([new_ids[i].item()],
                                                skip_special_tokens=True)
                    stream_callback(tok)

            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            return text, prompt_len, len(new_ids)

    def chat_generate(self, messages: list[dict], max_tokens: int = 256,
                      temperature: float = 0.7, top_p: float = 1.0,
                      stop: list[str] | None = None,
                      stream_callback=None) -> tuple[str, int, int]:
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        return self.generate(prompt, max_tokens, temperature, top_p,
                             stop, stream_callback)


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
            def event_stream():
                chunks = []
                def cb(tok: str):
                    chunks.append(tok)

                text, prompt_tokens, completion_tokens = generator.chat_generate(
                    messages, max_tokens=req.max_tokens,
                    temperature=req.temperature, top_p=req.top_p,
                    stop=req.stop, stream_callback=cb,
                )
                for i, tok in enumerate(chunks):
                    choice = Choice(
                        index=0,
                        delta={"role": "assistant" if i == 0 else None,
                               "content": tok},
                        finish_reason="stop" if i == len(chunks) - 1 else None,
                    )
                    chunk = {
                        "id": req_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": generator.model_name,
                        "choices": [choice.model_dump(exclude_none=True)],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        text, prompt_tokens, completion_tokens = generator.chat_generate(
            messages, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p, stop=req.stop,
        )
        return ChatCompletionResponse(
            id=req_id, created=int(time.time()), model=generator.model_name,
            choices=[Choice(index=0, message={"role": "assistant",
                                              "content": text},
                            finish_reason="stop")],
            usage=Usage(prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens),
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"

        if req.stream:
            def event_stream():
                chunks = []
                def cb(tok):
                    chunks.append(tok)
                generator.generate(req.prompt, max_tokens=req.max_tokens,
                                   temperature=req.temperature, top_p=req.top_p,
                                   stop=req.stop, stream_callback=cb)
                for i, tok in enumerate(chunks):
                    choice = Choice(index=0, text=tok,
                                    finish_reason="stop" if i == len(chunks) - 1 else None)
                    chunk = {
                        "id": req_id, "object": "text_completion.chunk",
                        "created": int(time.time()), "model": generator.model_name,
                        "choices": [choice.model_dump(exclude_none=True)],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")

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
