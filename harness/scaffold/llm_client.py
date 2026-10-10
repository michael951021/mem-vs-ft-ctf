"""LLM client for the agent scaffold.

Talks to the local Ollama endpoint (default) via its native /api/chat. Handles
reasoning models that emit <think>...</think> blocks: the visible answer is the
text AFTER the think block, but token accounting counts everything the model
produced (think tokens are real compute and must be billed to the budget).

Frozen base policy for arms A-D. Decoding params are fixed and logged.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field, asdict

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)  # unterminated think block


@dataclass
class LLMConfig:
    model: str = "qwen-agent:latest"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    num_ctx: int = 16384          # context window to request from Ollama
    num_predict: int = 2048       # max tokens per reply (room for <think>)
    keep_alive: str = "30m"
    timeout_s: int = 600
    think: bool = False           # reasoning-model <think>; OFF by default so
                                  # generations stay short under GPU contention.
                                  # Fixed+logged decoding choice, identical across arms.


@dataclass
class LLMResponse:
    text: str                     # visible answer (think block stripped)
    raw: str                      # full generation incl. think block
    thinking: str                 # extracted think content ("" if none)
    input_tokens: int
    output_tokens: int
    wall_clock_s: float
    load_s: float
    truncated: bool               # hit num_predict ceiling


def split_thinking(raw: str) -> tuple[str, str]:
    """Return (visible, thinking). Handles closed and unterminated <think>."""
    thinking_parts = THINK_RE.findall(raw)
    visible = THINK_RE.sub("", raw)
    thinking = "\n".join(thinking_parts)
    if "<think>" in visible:  # unterminated (ran out of tokens mid-think)
        m = OPEN_THINK_RE.search(visible)
        if m:
            thinking = (thinking + "\n" + m.group(0)).strip()
            visible = OPEN_THINK_RE.sub("", visible)
    return visible.strip(), thinking.strip()


class OllamaClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def chat(self, messages: list[dict], stop: list[str] | None = None) -> LLMResponse:
        """messages: [{"role": "system|user|assistant", "content": "..."}]."""
        cfg = self.cfg
        options = {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "seed": cfg.seed,
            "num_ctx": cfg.num_ctx,
            "num_predict": cfg.num_predict,
        }
        if stop:
            options["stop"] = stop
        payload = {
            "model": cfg.model,
            "messages": messages,
            "stream": False,
            "keep_alive": cfg.keep_alive,
            "think": cfg.think,
            "options": options,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{cfg.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
            body = json.loads(r.read().decode())
        wall = time.time() - t0

        raw = body.get("message", {}).get("content", "")
        visible, thinking = split_thinking(raw)
        out_tok = int(body.get("eval_count", 0) or 0)
        return LLMResponse(
            text=visible,
            raw=raw,
            thinking=thinking,
            input_tokens=int(body.get("prompt_eval_count", 0) or 0),
            output_tokens=out_tok,
            wall_clock_s=wall,
            load_s=round((body.get("load_duration", 0) or 0) / 1e9, 3),
            truncated=(out_tok >= cfg.num_predict),
        )


if __name__ == "__main__":
    import sys
    cfg = LLMConfig(model=sys.argv[1] if len(sys.argv) > 1 else "qwen-agent:latest")
    client = OllamaClient(cfg)
    r = client.chat([
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "In one short sentence: what is a CTF flag format like picoCTF?"},
    ])
    print("VISIBLE:", r.text)
    print("THINKING chars:", len(r.thinking))
    print("tokens in/out:", r.input_tokens, r.output_tokens, "truncated:", r.truncated)
    print("wall_s:", round(r.wall_clock_s, 2), "load_s:", r.load_s)
