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

try:
    from PIL import Image
    import io, base64, re as _re
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ── OpenAI-compatible request/response schemas ──────────────────────────

class ImageURL(BaseModel):
    url: str

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

class Message(BaseModel):
    role: str
    content: str | list[ContentPart]

class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    max_tokens: int = Field(default=256, alias="max_tokens")
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[list[str]] = None
    enable_thinking: bool = False
    detail: Optional[str] = None

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
                 dtype: str = "float16", max_seq_len: int = 4096,
                 compression: str | None = None,
                 prefetching: bool = False):
        from airllm import AutoModel
        self.model_path = model_path
        self.device = device
        _DTYPE_ALIASES = {
            "float8": "float8_e4m3fn",
            "fp8": "float8_e4m3fn",
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }
        dtype_key = _DTYPE_ALIASES.get(dtype, dtype)
        self._dtype = getattr(torch, dtype_key)
        self.max_seq_len = max_seq_len
        self.model = AutoModel.from_pretrained(
            model_path, device=device, dtype=self._dtype,
            max_seq_len=max_seq_len, compression=compression,
            prefetching=prefetching,
        )
        self.tokenizer = self.model.tokenizer
        if self.is_multimodal and hasattr(self.tokenizer, 'tokenizer'):
            self._text_tokenizer = self.tokenizer.tokenizer
        else:
            self._text_tokenizer = self.tokenizer
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return os.path.basename(self.model_path.rstrip("/"))

    @property
    def is_multimodal(self) -> bool:
        return hasattr(self.model, '_is_vl_model') and self.model._is_vl_model

    @staticmethod
    def _load_image_from_url(url: str) -> "Image.Image":
        if not _PIL_AVAILABLE:
            raise ImportError("PIL is required for image inputs: pip install Pillow")
        if url.startswith("data:"):
            match = _re.match(r"data:image/[^;]+;base64,(.*)", url)
            if match:
                data = base64.b64decode(match.group(1))
                return Image.open(io.BytesIO(data)).convert("RGB")
        if url.startswith("http://") or url.startswith("https://"):
            import requests
            resp = requests.get(url, timeout=30)
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        return Image.open(url).convert("RGB")

    def _extract_images_from_messages(self, messages: list[dict]) -> list["Image.Image"]:
        images = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url:
                            images.append(self._load_image_from_url(url))
        return images

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
                ids = self._text_tokenizer(s, add_special_tokens=False).input_ids
                stop_ids.extend(ids)
            if stop_ids:
                kwargs["eos_token_id"] = list(set(
                    [self._text_tokenizer.eos_token_id] + stop_ids
                ))
        return kwargs

    def generate(self, prompt: str = None, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 1.0,
                 stop: list[str] | None = None,
                 pixel_values=None, image_grid_thw=None,
                 input_ids=None) -> tuple[str, int, int]:
        with self._lock:
            if input_ids is None:
                tok = self._text_tokenizer if pixel_values is None else self.tokenizer
                inputs = tok(prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device)
            else:
                input_ids = input_ids.to(self.device)
            prompt_len = input_ids.shape[1]
            gen_kwargs = self._prepare_kwargs(max_tokens, temperature, top_p, stop)
            if pixel_values is not None:
                gen_kwargs["pixel_values"] = pixel_values.to(
                    device=self.device, dtype=self.model.running_dtype
                )
                gen_kwargs["image_grid_thw"] = image_grid_thw
            output_ids = self.model.generate(input_ids, **gen_kwargs)
            new_ids = output_ids[0, prompt_len:]
            text = self._text_tokenizer.decode(new_ids, skip_special_tokens=True)
            return text, prompt_len, len(new_ids)

    def generate_stream(self, prompt: str = None, max_tokens: int = 256,
                        temperature: float = 0.7, top_p: float = 1.0,
                        stop: list[str] | None = None,
                        pixel_values=None, image_grid_thw=None,
                        input_ids=None) -> Generator[str, None, None]:
        """True streaming: yields text chunks as they are generated."""
        with self._lock:
            if input_ids is None:
                tok = self._text_tokenizer if pixel_values is None else self.tokenizer
                inputs = tok(prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device)
            else:
                input_ids = input_ids.to(self.device)
            prompt_len = input_ids.shape[1]

            eos_ids = {self._text_tokenizer.eos_token_id}
            if stop:
                for s in stop:
                    ids = self._text_tokenizer(s, add_special_tokens=False).input_ids
                    for tid in ids:
                        eos_ids.add(tid)

            logger.info(f"Stream generate: prompt_len={prompt_len}, max_new={max_tokens}")

            past_kv = None
            generated_ids = []
            prev_text = ""

            fwd_kwargs = {"use_cache": True}
            if pixel_values is not None:
                fwd_kwargs["pixel_values"] = pixel_values.to(
                    device=self.device, dtype=self.model.running_dtype
                )
                fwd_kwargs["image_grid_thw"] = image_grid_thw

            outputs = self.model(input_ids=input_ids, **fwd_kwargs)
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
                full_text = self._text_tokenizer.decode(generated_ids,
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
        images = self._extract_images_from_messages(messages) if self.is_multimodal else []
        prompt, pre_input_ids, pixel_values, image_grid_thw = self._build_chat_prompt(
            messages, images=images, enable_thinking=enable_thinking,
        )
        text, prompt_tokens, completion_tokens = self.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, stop=stop,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            input_ids=pre_input_ids,
        )
        if enable_thinking:
            content, reasoning = self._parse_thinking_content(text)
            return content, prompt_tokens, completion_tokens, reasoning
        return text, prompt_tokens, completion_tokens, None

    def chat_generate_stream(self, messages: list[dict], max_tokens: int = 256,
                             temperature: float = 0.7, top_p: float = 1.0,
                             stop: list[str] | None = None,
                             enable_thinking: bool = False) -> Generator[tuple[str, Optional[str]], None, None]:
        images = self._extract_images_from_messages(messages) if self.is_multimodal else []
        prompt, pre_input_ids, pixel_values, image_grid_thw = self._build_chat_prompt(
            messages, images=images, enable_thinking=enable_thinking,
        )
        thinking_buffer = ""
        think_open = False
        think_closed = False
        for chunk in self.generate_stream(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, stop=stop,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            input_ids=pre_input_ids,
        ):
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

    def _build_chat_prompt(self, messages: list[dict], images: list | None = None,
                           enable_thinking: bool = False) -> tuple:
        text_messages = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            parts.append("<|image_pad|>")
                text_messages.append({"role": m["role"], "content": " ".join(parts) if parts else ""})
            else:
                text_messages.append({"role": m["role"], "content": content or ""})

        pixel_values = None
        image_grid_thw = None

        if images and self.is_multimodal:
            try:
                has_processor = hasattr(self.tokenizer, 'image_processor') or hasattr(self.tokenizer, 'process_images')
                if has_processor:
                    text = self.tokenizer.apply_chat_template(
                        text_messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                    proc_inputs = self.tokenizer(
                        text=[text],
                        images=images,
                        return_tensors="pt",
                    )
                    input_ids = proc_inputs["input_ids"]
                    if "pixel_values" in proc_inputs:
                        pixel_values = proc_inputs["pixel_values"]
                    if "image_grid_thw" in proc_inputs:
                        image_grid_thw = proc_inputs["image_grid_thw"]
                    return None, input_ids, pixel_values, image_grid_thw
            except Exception as e:
                logger.warning(f"Processor-based image handling failed: {e}, falling back to text-only")

        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                text_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in text_messages)

        return prompt, None, pixel_values, image_grid_thw


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
        messages = []
        for m in req.messages:
            if isinstance(m.content, list):
                content_list = []
                for part in m.content:
                    if part.type == "text" and part.text:
                        content_list.append({"type": "text", "text": part.text})
                    elif part.type == "image_url" and part.image_url:
                        content_list.append({"type": "image_url", "image_url": {"url": part.image_url.url}})
                messages.append({"role": m.role, "content": content_list})
            else:
                messages.append({"role": m.role, "content": m.content or ""})
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
    parser.add_argument("--compression", type=str, default=None,
                        help="Weight compression: '4bit' or '8bit' (requires bitsandbytes)")
    parser.add_argument("--prefetching", action="store_true", default=False,
                        help="Overlap disk I/O with GPU compute during layer loading")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    generator = AirLLMGenerator(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
        compression=args.compression,
        prefetching=args.prefetching,
    )
    print(f"Model '{generator.model_name}' loaded. Starting server...")

    app = create_app(generator)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
