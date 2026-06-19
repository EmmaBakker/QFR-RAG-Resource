from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_session = requests.Session()


@dataclass(frozen=True)
class LLMConfig:
    """
    backends:
    - "ollama": native Ollama via langchain_ollama.ChatOllama
    - "openai": standard OpenAI API (api.openai.com); api_key required
    - "openai_compat": OpenAI-compatible server (vLLM, LM Studio); base_url required
    - "anthropic": Claude API (api.anthropic.com); api_key required
    """    
    backend: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_s: int = 600
    max_retries: int = 2
    retry_backoff_s: float = 1.5
    system_message: Optional[str] = None
    num_ctx: Optional[int] = None
    keep_alive: Optional[int] = None
    format: Optional[str] = None
    num_gpu: Optional[int] = None


def _safe_llm_meta(llm_cfg: "LLMConfig") -> Dict[str, Any]:
    """Return a JSON-serialisable dict of ``llm_cfg`` with the API key redacted.

    Use this instead of ``vars(llm_cfg)`` when writing ``meta.json`` files so
    that credentials are never accidentally committed or logged.
    """
    meta = dict(vars(llm_cfg))
    if meta.get("api_key"):
        meta["api_key"] = "<redacted>"
    return meta


def _sleep_backoff(i: int, base: float) -> None:
    time.sleep(base * (2 ** i))


@lru_cache(maxsize=8)
def _get_ollama_client(
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    num_ctx: Optional[int],
    keep_alive: Optional[int],
    format: Optional[str],
    num_gpu: Optional[int] = None,
):
    from langchain_ollama import ChatOllama  # type: ignore

    kwargs: Dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    if num_ctx is not None:
        kwargs["num_ctx"] = num_ctx
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    if format is not None:
        kwargs["format"] = format
    if num_gpu is not None:
        kwargs["num_gpu"] = num_gpu
    return ChatOllama(**kwargs)


@lru_cache(maxsize=8)
def _get_openai_client(backend: str, api_key: Optional[str], base_url: Optional[str]):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError("pip install openai") from e
    if backend == "openai":
        return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    if backend == "openai_compat":
        if not base_url:
            raise RuntimeError("openai_compat requires base_url.")
        return OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
    raise ValueError(
        f"Unknown backend: {backend!r}. Allowed: ollama, openai, openai_compat, anthropic"
    )


@lru_cache(maxsize=4)
def _get_anthropic_client(api_key: Optional[str]):
    try:
        from anthropic import Anthropic  # type: ignore
    except Exception as e:
        raise ImportError("pip install anthropic") from e
    return Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))


def _build_lc_messages(cfg: LLMConfig, prompt: str):
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    msgs = []
    if cfg.system_message:
        msgs.append(SystemMessage(content=cfg.system_message))
    msgs.append(HumanMessage(content=prompt))
    return msgs


def _extract_ollama_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "eval_count": meta.get("eval_count"),
        "prompt_eval_count": meta.get("prompt_eval_count"),
        "eval_duration_ms": round((meta.get("eval_duration") or 0) / 1e6, 1),
        "prompt_eval_duration_ms": round((meta.get("prompt_eval_duration") or 0) / 1e6, 1),
        "load_duration_ms": round((meta.get("load_duration") or 0) / 1e6, 1),
        "total_duration_ms": round((meta.get("total_duration") or 0) / 1e6, 1),
    }


def stream_generate_with_info(
    prompt: str, cfg: LLMConfig
) -> Tuple[Iterator[str], Dict[str, Any]]:
    """Return (token_iterator, info_dict)."""
    info: Dict[str, Any] = {"backend": cfg.backend, "model": cfg.model}

    # -- Ollama --
    if cfg.backend == "ollama":
        llm = _get_ollama_client(
            base_url=cfg.base_url or "",
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            num_ctx=cfg.num_ctx,
            keep_alive=cfg.keep_alive,
            format=cfg.format,
            num_gpu=cfg.num_gpu,
        )
        messages = _build_lc_messages(cfg, prompt)

        def _gen() -> Iterator[str]:
            t0 = time.time()
            last_meta: Dict[str, Any] = {}
            try:
                for chunk in llm.stream(messages):
                    content = chunk.content or ""
                    if content:
                        yield content
                    if chunk.response_metadata:
                        last_meta = chunk.response_metadata
            except Exception as exc:
                info["error"] = str(exc)
                raise
            finally:
                info["latency_s"] = time.time() - t0
                if last_meta:
                    info.update(_extract_ollama_meta(last_meta))

        return _gen(), info

    # -- Anthropic --
    if cfg.backend == "anthropic":
        client = _get_anthropic_client(cfg.api_key)
        # Claude Messages API: system + user messages
        msgs: List[Dict[str, Any]] = []
        if cfg.system_message:
            msgs.append({"role": "system", "content": cfg.system_message})
        msgs.append({"role": "user", "content": prompt})

        last_err = None
        t0 = time.time()
        for i in range(cfg.max_retries + 1):
            try:
                resp = client.messages.create(
                    model=cfg.model,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    messages=msgs,
                    timeout=cfg.timeout_s,
                )
                # Treat the whole response text as one chunk (no streaming)
                parts = resp.content or []
                text = "".join(
                    p.text for p in parts if getattr(p, "type", None) == "text"
                )
                info["latency_s"] = time.time() - t0
                info["retries_used"] = i
                return iter([text]), info
            except Exception as e:
                last_err = e
                if i < cfg.max_retries:
                    _sleep_backoff(i, cfg.retry_backoff_s)

        info["latency_s"] = time.time() - t0
        info["error"] = str(last_err)
        raise RuntimeError(
            f"LLM request failed. backend={cfg.backend} model={cfg.model} error={last_err}"
        )

    # -- OpenAI / OpenAI-compat --
    client = _get_openai_client(cfg.backend, cfg.api_key, cfg.base_url)
    oa_messages: List[Dict[str, Any]] = []
    if cfg.system_message:
        oa_messages.append({"role": "system", "content": cfg.system_message})
    oa_messages.append({"role": "user", "content": prompt})

    last_err = None
    t0 = time.time()
    for i in range(cfg.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=oa_messages,
                temperature=float(cfg.temperature),
                max_tokens=int(cfg.max_tokens),
                timeout=cfg.timeout_s,
            )
            text = resp.choices[0].message.content or ""
            info["latency_s"] = time.time() - t0
            info["retries_used"] = i
            return iter([text]), info
        except Exception as e:
            last_err = e
            if i < cfg.max_retries:
                _sleep_backoff(i, cfg.retry_backoff_s)

    info["latency_s"] = time.time() - t0
    info["error"] = str(last_err)
    raise RuntimeError(
        f"LLM request failed. backend={cfg.backend} model={cfg.model} error={last_err}"
    )


def generate_with_info(prompt: str, cfg: LLMConfig) -> Tuple[str, Dict[str, Any]]:
    token_iter, info = stream_generate_with_info(prompt, cfg)
    text = "".join(token_iter)
    return text, info


def generate(prompt: str, cfg: LLMConfig) -> str:
    text, _ = generate_with_info(prompt, cfg)
    return text