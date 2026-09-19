#!/usr/bin/env python3
"""
agent-demo.py — a small, transparent tool-using agent on the local llama-server (:4791, OpenAI-compatible, --jinja).
Usage: agent-demo.py "task"   (or no argument: interactive — type tasks, empty line quits)

Tools (all inside ~/agent-sandbox): run_shell, write_file, read_file, list_dir. Every step is printed live:
what the model says, which tool it calls with which arguments, what came back, and the speed of each step.
"""
import json, os, subprocess, sys, time, urllib.request, shlex

URL = os.environ.get("AGENT_URL", "http://127.0.0.1:4791/v1/chat/completions")
SANDBOX = os.path.expanduser(os.environ.get("AGENT_SANDBOX", "~/agent-sandbox"))
os.makedirs(SANDBOX, exist_ok=True)
MAX_STEPS = int(os.environ.get("AGENT_STEPS", "14"))
TEMP = float(os.environ.get("AGENT_TEMP", "0.3"))

C = {"dim": "\033[2m", "b": "\033[1m", "cyan": "\033[36m", "yel": "\033[33m", "grn": "\033[32m", "red": "\033[31m", "mag": "\033[35m", "0": "\033[0m"}
def p(color, s): print(C[color] + s + C["0"], flush=True)

def _safe(path):
    full = os.path.realpath(os.path.join(SANDBOX, path))
    if not full.startswith(os.path.realpath(SANDBOX) + os.sep) and full != os.path.realpath(SANDBOX):
        raise ValueError("path outside the sandbox")
    return full

BLOCK = ("rm -rf /", "mkfs", "dd if=", ":(){", "shutdown", "reboot", "sudo ")
def run_shell(command: str):
    if any(b in command for b in BLOCK): return "refused: dangerous command"
    try:
        r = subprocess.run(command, shell=True, cwd=SANDBOX, capture_output=True, text=True, timeout=90)
        out = (r.stdout + (("\n[stderr] " + r.stderr) if r.stderr.strip() else "")).strip()
        return (out or "(no output)")[:4000] + (f"\n[exit {r.returncode}]" if r.returncode else "")
    except subprocess.TimeoutExpired: return "timeout after 90 s"
def write_file(path: str, content: str):
    full = _safe(path); os.makedirs(os.path.dirname(full) or SANDBOX, exist_ok=True)
    open(full, "w", encoding="utf-8").write(content); return f"wrote {len(content)} chars to {path}"
def read_file(path: str):
    return open(_safe(path), encoding="utf-8", errors="replace").read()[:6000]
def list_dir(path: str = "."):
    return "\n".join(sorted(os.listdir(_safe(path)))) or "(empty)"

TOOLS = [
 {"type": "function", "function": {"name": "run_shell", "description": "Run a bash command inside the sandbox directory and return its output (stdout+stderr). Use it to run scripts, python3, list files, check results.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
 {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a text file (path relative to the sandbox).", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
 {"type": "function", "function": {"name": "read_file", "description": "Read a text file (path relative to the sandbox).", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
 {"type": "function", "function": {"name": "list_dir", "description": "List files in a sandbox directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
]
IMPL = {"run_shell": run_shell, "write_file": write_file, "read_file": read_file, "list_dir": list_dir}

SYSTEM = ("You are a capable local agent working inside a sandbox directory. Use the tools to actually do the task: write files, "
          "run commands, read outputs, verify results, and fix errors yourself. Call one tool at a time and wait for its result. "
          "When the task is fully done and verified, reply with a short final summary (no tool call). Answer in the user's language.")

def chat(messages):
    body = {"messages": messages, "tools": TOOLS, "tool_choice": "auto", "temperature": TEMP, "max_tokens": 700}
    if os.environ.get("AGENT_THINK") == "1": body["chat_template_kwargs"] = {"enable_thinking": True}; body["max_tokens"] = 1500
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t0
    return d["choices"][0]["message"], d.get("timings", {}), dt

def run_task(task):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    p("b", f"\n▶ المهمة: {task}\n" + "─" * 70)
    t_start = time.time(); total_tok = 0
    for step in range(1, MAX_STEPS + 1):
        msg, tm, dt = chat(messages)
        total_tok += tm.get("predicted_n", 0)
        speed = f"{tm.get('predicted_per_second', 0):.1f} توكن/ث" if tm else ""
        content = (msg.get("content") or "").strip()
        calls = msg.get("tool_calls") or []
        if content: p("cyan", f"[{step}] 🧠 {content[:1200]}")
        if not calls:
            p("grn", f"\n✅ انتهى في {step} خطوات، {total_tok} توكن، {time.time()-t_start:.0f} ث  ({speed})"); return
        messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        for call in calls:
            fn = call["function"]["name"]; raw = call["function"].get("arguments") or "{}"
            try: args = json.loads(raw) if isinstance(raw, str) else raw
            except Exception: args = {"command": raw} if fn == "run_shell" else {}
            p("yel", f"[{step}] 🔧 {fn}({', '.join(f'{k}={shlex.quote(str(v))[:160]}' for k, v in args.items())})   {C['dim']}{speed}{C['0']}")
            try: result = IMPL[fn](**args) if fn in IMPL else f"unknown tool {fn}"
            except Exception as e: result = f"error: {e}"
            p("mag", "    ↳ " + str(result)[:900].replace("\n", "\n      "))
            messages.append({"role": "tool", "tool_call_id": call.get("id", fn), "name": fn, "content": str(result)})
    p("red", f"\n⛔ توقف بعد {MAX_STEPS} خطوة")

if __name__ == "__main__":
    p("dim", f"sandbox: {SANDBOX}   server: {URL}")
    if len(sys.argv) > 1:
        run_task(" ".join(sys.argv[1:]))
    while True:
        try: task = input(C["b"] + "\nاكتب مهمة (Enter فارغ للخروج): " + C["0"]).strip()
        except EOFError: break
        if not task: break
        run_task(task)
