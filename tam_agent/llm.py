"""LLM client with native tool calling. Pick a provider with environment variables (see .env.example).

  LLM_PROVIDER=ollama     free, local (https://ollama.com) - use a tool-calling model, e.g. qwen2.5:7b, llama3.1:8b
  LLM_PROVIDER=anthropic  needs ANTHROPIC_API_KEY and `pip install anthropic`
  LLM_PROVIDER=openai     needs OPENAI_API_KEY and `pip install openai`

The agent talks to `chat_tools()` with a provider-neutral message list:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
and gets back {"text": str, "tool_calls": [{"id", "name", "args"}]}.
"""
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

# minimal .env loader (no extra dependency)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

DEFAULT_MODELS = {"ollama": "qwen2.5:7b", "anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini"}


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        self.calls = 0  # simple counter reported in run_log for cost/latency
        self.input_tokens = 0
        self.output_tokens = 0
        self._client = None

    # ------------------------------------------------------------ plain chat (kept for compatibility)
    def chat(self, messages: list[dict], system: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        return self.chat_tools(messages, [], system, max_tokens, temperature)["text"]

    # ------------------------------------------------------------ tool calling
    def chat_tools(self, messages: list[dict], tools: list[dict], system: str | None = None,
                   max_tokens: int = 1500, temperature: float = 0.0) -> dict:
        self.calls += 1
        fn = {"anthropic": self._anthropic, "openai": self._openai, "ollama": self._ollama}[self.provider]
        for attempt in range(3):  # retry transient API errors
            try:
                return fn(messages, tools, system, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001
                if attempt == 2 or isinstance(e, (ValueError, KeyError)):
                    raise
                time.sleep(2 ** attempt * 2)

    def _anthropic(self, messages, tools, system, max_tokens, temperature):
        import anthropic
        self._client = self._client or anthropic.Anthropic()
        msgs = []
        for m in messages:
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(block)
                else:
                    msgs.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                           for c in m.get("tool_calls", [])]
                msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no content)"}]})
            elif msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                msgs[-1]["content"].append({"type": "text", "text": m["content"]})  # keep roles alternating
            else:
                msgs.append({"role": "user", "content": m["content"]})
        kw = {"tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                        for t in tools]} if tools else {}
        resp = self._client.messages.create(model=self.model, max_tokens=max_tokens, temperature=temperature,
                                            system=system or "", messages=msgs, **kw)
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "args": b.input or {}} for b in resp.content if b.type == "tool_use"]
        return {"text": text, "tool_calls": calls}

    def _openai(self, messages, tools, system, max_tokens, temperature):
        from openai import OpenAI
        self._client = self._client or OpenAI()
        msgs = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "tool":
                msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            elif m["role"] == "assistant":
                d = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    d["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                       for c in m["tool_calls"]]
                msgs.append(d)
            else:
                msgs.append({"role": "user", "content": m["content"]})
        kw = {"tools": [{"type": "function", "function": t} for t in tools]} if tools else {}
        resp = self._client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens,
                                                    temperature=temperature, **kw)
        if resp.usage:
            self.input_tokens += resp.usage.prompt_tokens
            self.output_tokens += resp.usage.completion_tokens
        msg = resp.choices[0].message
        calls = []
        for c in msg.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": c.function.arguments}
            calls.append({"id": c.id, "name": c.function.name, "args": args})
        return {"text": msg.content or "", "tool_calls": calls}

    def _ollama(self, messages, tools, system, max_tokens, temperature):
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        msgs = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "tool":
                msgs.append({"role": "tool", "content": m["content"], "tool_name": m["name"]})
            elif m["role"] == "assistant":
                d = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    d["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["args"]}}
                                       for c in m["tool_calls"]]
                msgs.append(d)
            else:
                msgs.append({"role": "user", "content": m["content"]})
        body = {"model": self.model, "messages": msgs, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens,
                            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384"))}}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        self.input_tokens += data.get("prompt_eval_count", 0)
        self.output_tokens += data.get("eval_count", 0)
        msg = data["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            args = c["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "name": c["function"]["name"], "args": args})
        return {"text": msg.get("content") or "", "tool_calls": calls}
