#!/usr/bin/env python3
"""
sections-proxy.py — OpenAI-compatible proxy in front of llama-server that turns the machine's BATCH throughput
into answer speed (2026-09-19).

Why: on this CPU the model is memory-bound. One weight sweep (~250 ms) advances ONE sequence by one token, but the
same sweep can advance 8-16 sequences at ~55 ms/token of extra compute each. A single stream therefore tops out at
~7 tok/s (with MTP), while 8-16 sequences together reach 11-18 tok/s aggregate. This proxy makes one answer BE
several sequences: it asks the model for a skeleton (section titles), generates every section concurrently in its
own llama-server slot, and streams the sections back in order as one continuous answer. The model, its weights and
its sampling are untouched; each section is an ordinary exact generation.

Flow per streaming chat request:
  1. If the server has 1 slot, or the request looks short/unstructured, pass through unchanged (MTP single stream).
  2. Skeleton call (greedy, <=120 tokens): numbered list of 4-N section titles, or the word SINGLE.
  3. N concurrent section generations (same conversation + a system instruction restricting to section k).
  4. Stream section 1 live; the others are buffered while they run and flushed in order when their turn comes.

Env: UPSTREAM (default http://127.0.0.1:4791), PORT (4792), SECTIONS_MAX (16), SECTIONS_MIN_WORDS (8),
SECTIONS_TOKENS (section max_tokens, default 350), SECTIONS_DISABLE=1 (pure pass-through), LOG (/tmp/sections-proxy.log).
"""
import http.client, http.server, json, os, re, socketserver, sys, threading, time, urllib.parse, uuid

UP = os.environ.get("UPSTREAM", "http://127.0.0.1:4791")
PORT = int(os.environ.get("PORT", "4792"))
SECTIONS_MAX = int(os.environ.get("SECTIONS_MAX", "16"))
SECTIONS_MIN_WORDS = int(os.environ.get("SECTIONS_MIN_WORDS", "12"))
SECTIONS_TOKENS = int(os.environ.get("SECTIONS_TOKENS", "350"))
DISABLE = os.environ.get("SECTIONS_DISABLE") == "1"
LOG = os.environ.get("LOG", "/tmp/sections-proxy.log")
_u = urllib.parse.urlparse(UP); UP_HOST, UP_PORT = _u.hostname, _u.port

def log(msg):
    try:
        with open(LOG, "a") as f: f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    except Exception: pass

def upstream(method, path, body=None, headers=None, timeout=900):
    c = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=timeout)
    h = {"Content-Type": "application/json"}
    if headers: h.update(headers)
    c.request(method, path, body=body, headers=h)
    return c, c.getresponse()

def slot_count():
    try:
        c, r = upstream("GET", "/slots", timeout=5); n = len(json.loads(r.read())); c.close(); return n
    except Exception: return 1

# ---------------------------------------------------------------- skeleton
SKELETON_SYS = (
    "Before answering: plan the STRUCTURE of the answer to the message above, do not write the answer itself. "
    "Reply with ONLY a numbered list of 4 to {n} short section titles (one line each, no text after the list) that together "
    "fully answer the request, in the user's language. If the request is better answered as ONE continuous piece "
    "(a short answer, a single function, a story, a proof, a translation, a yes/no question, casual chat), reply with exactly the single word: SINGLE"
)
STRUCT_HINT = re.compile(r"(اشرح|اكتب|قائمة|خطة|خطوات|قارن|طرق|أسباب|نصائح|فوائد|أنواع|explain|write|list|plan|steps|compare|ways|reasons|tips|benefits|types|guide|how to|describe|overview|summar)", re.I)

def wants_sections(messages):
    last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not last: return False
    text = last.get("content") if isinstance(last.get("content"), str) else " ".join(p.get("text", "") for p in (last.get("content") or []) if isinstance(p, dict))
    words = len(text.split())
    return words >= SECTIONS_MIN_WORDS or bool(STRUCT_HINT.search(text))

def parse_skeleton(text):
    if "SINGLE" in text.upper()[:40]: return []
    titles = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*]|\(?\d+[\).:\-٠-٩]*)\s*(.+?)\s*$", line)
        if m:
            t = re.sub(r"[*_`#]+", "", m.group(1)).strip(" :.-")
            if 2 <= len(t) <= 120: titles.append(t)
    # de-dup, cap
    seen, out = set(), []
    for t in titles:
        k = t.lower()
        if k not in seen: seen.add(k); out.append(t)
    return out[:SECTIONS_MAX]


def with_instruction(messages, text):
    """Copy of messages with `text` appended to the LAST user message. The rendered prompt then shares its whole
    prefix (everything before that message) with the skeleton call and with every other section, so llama-server's
    prompt cache (--cache-ram) restores the conversation state into each slot and only the tail is recomputed."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                m["content"] = c + [{"type": "text", "text": "\n\n" + text}]
            else:
                m["content"] = (c or "") + "\n\n" + text
            break
    return out

PLAN_INLINE = (
    "[Instruction] Reply format: line 1 must be `PLAN: t1 | t2 | ... | tN` = 4-{n} short section titles (user's language) that "
    "together answer the message above; explanations, lists, how-tos, comparisons and plans always get sections. Only a short factual "
    "answer, casual chat, a single function, a story or a proof gets `PLAN: SINGLE` followed by the full answer. After a multi-title "
    "PLAN write ONLY section 1 under `## t1` and stop (the other sections are produced in parallel)."
)

def parse_plan_line(line):
    m = re.search(r"PLAN\s*:\s*(.+)", line)
    if not m: return None
    body = m.group(1).strip()
    if body.upper().startswith("SINGLE"): return []
    parts = [re.sub(r"[*_`#]+", "", t).strip(" :.-") for t in re.split(r"\s*[|؛;]\s*", body)]
    parts = [t for t in parts if 2 <= len(t) <= 120]
    seen, out = set(), []
    for t in parts:
        if t.lower() not in seen: seen.add(t.lower()); out.append(t)
    return out[:SECTIONS_MAX]

def get_skeleton(req, n_max):
    body = dict(req); body.pop("stream", None); body.pop("stream_options", None)
    body["messages"] = with_instruction(req["messages"], "[Instruction] " + SKELETON_SYS.format(n=n_max))
    body["max_tokens"] = 120; body["temperature"] = 0
    c, r = upstream("POST", "/v1/chat/completions", json.dumps(body))
    d = json.loads(r.read()); c.close()
    text = (d.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return parse_skeleton(text), text

# ---------------------------------------------------------------- sections
def section_system(k, n, title, titles):
    others = " | ".join(f"{i+1}. {t}" for i, t in enumerate(titles))
    return (f"[Instruction] Answer plan: {others}. Write ONLY section {k} `## {title}` in the user's language, complete but concise, "
            f"then STOP (no other sections, no intro, no conclusion).")

class Section:
    def __init__(self, idx): self.idx = idx; self.parts = []; self.done = False; self.err = None; self.cv = threading.Condition(); self.tokens = 0; self.emitted = ""; self.cut = False

def run_section(req, sec, k, n, titles, max_tokens, instruction=None):
    body = dict(req); body.pop("stream_options", None)
    body["messages"] = req["messages"] if instruction == "" else with_instruction(req["messages"], instruction if instruction is not None else section_system(k, n, titles[k-1], titles))
    if os.environ.get("SECTIONS_NO_SPEC") == "1": body["speculative.n_max"] = 0
    body["stream"] = True; body["max_tokens"] = max_tokens; body["cache_prompt"] = True
    try:
        c, r = upstream("POST", "/v1/chat/completions", json.dumps(body))
        buf = b""
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk: break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1); line = line.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]": continue
                try: d = json.loads(line[6:])
                except Exception: continue
                ch = (d.get("choices") or [{}])[0]; delta = (ch.get("delta") or {}).get("content") or ""
                if delta:
                    with sec.cv: sec.parts.append(delta); sec.tokens += 1; sec.cv.notify_all()
        c.close()
    except Exception as e:
        sec.err = str(e)
    with sec.cv: sec.done = True; sec.cv.notify_all()

# ---------------------------------------------------------------- HTTP
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # close-delimited streaming; no Content-Length bookkeeping
    def log_message(self, *a): pass

    def _passthrough(self, method, body):
        try:
            hdrs = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "authorization", "accept")}
            c, r = upstream(method, self.path, body, hdrs)
        except Exception as e:
            self.send_error(502, f"upstream: {e}"); return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() in ("content-type", "cache-control"): self.send_header(k, v)
        self.end_headers()
        try:
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk: break
                self.wfile.write(chunk); self.wfile.flush()
        except Exception: pass
        c.close()

    def _sse(self, rid, model, content=None, finish=None, extra=None):
        d = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
             "choices": [{"index": 0, "delta": ({"content": content} if content is not None else {}), "finish_reason": finish}]}
        if extra: d.update(extra)
        self.wfile.write(("data: " + json.dumps(d, ensure_ascii=False) + "\n\n").encode("utf-8")); self.wfile.flush()

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            # expose a second id: same model with thinking enabled per request
            try:
                c, r = upstream("GET", "/v1/models"); d = json.loads(r.read()); c.close()
                base = [m for m in d.get("data", [])]
                for m in list(base):
                    t = dict(m); t["id"] = m["id"] + "-thinking"; base.append(t)
                d["data"] = base; body = json.dumps(d).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body); return
            except Exception: pass
        self._passthrough("GET", None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else b""
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._passthrough("POST", body)
        try: req = json.loads(body or b"{}")
        except Exception: return self._passthrough("POST", body)
        if str(req.get("model", "")).endswith("-thinking"):
            # "<alias>-thinking": same model, thinking mode on for this request (server merges per-request kwargs)
            req["model"] = req["model"][:-len("-thinking")]
            req["chat_template_kwargs"] = {"enable_thinking": True}
            body = json.dumps(req).encode("utf-8")
        if DISABLE:
            return self._passthrough("POST", body)
        if not req.get("stream") or not req.get("messages") or not wants_sections(req["messages"]):
            return self._passthrough("POST", body)
        slots = slot_count()
        if slots < 3:
            return self._passthrough("POST", body)
        t0 = time.time()
        model = req.get("model", "bonsai-27b-uncensored"); rid = "chatcmpl-" + uuid.uuid4().hex[:12]
        max_tokens = int(req.get("max_tokens") or 0) or SECTIONS_TOKENS
        max_tokens = min(max_tokens, SECTIONS_TOKENS) if req.get("max_tokens") is None else max_tokens
        if os.environ.get("SECTIONS_INLINE", "1") == "1":
            # Inline plan: section 1 starts immediately and its first line is the plan; the other sections launch as
            # soon as that line has streamed (no separate 20 s skeleton call).
            n_max = min(slots, SECTIONS_MAX)
            first = Section(0)
            threading.Thread(target=run_section, args=(req, first, 1, n_max, [], max_tokens + 60, PLAN_INLINE.format(n=n_max)), daemon=True).start()
            plan_line = None
            while True:
                with first.cv:
                    while not first.done and "\n" not in "".join(first.parts): first.cv.wait(0.25)
                    joined = "".join(first.parts); done = first.done
                if "\n" in joined:
                    plan_line = joined.split("\n", 1)[0]; break
                if done: plan_line = joined; break
            titles = parse_plan_line(plan_line or "")
            t_sk = time.time() - t0
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
            self._sse(rid, model, "")
            if titles is None or len(titles) < 2:
                # SINGLE (or no plan line): relay this stream minus the PLAN line; if the model stopped right after
                # the PLAN line, run the request untouched instead so the user always gets a full answer.
                with first.cv:
                    while not first.done: first.cv.wait(0.5)
                    rest = "".join(first.parts).split("\n", 1)[1] if "\n" in "".join(first.parts) else ""
                if len(rest.strip()) >= 20:
                    log(f"inline plan -> SINGLE ({t_sk:.1f}s), relaying: {plan_line!r}")
                    self._relay_sections(rid, model, [first], t0, t_sk, strip_first_line=(titles is not None))
                else:
                    log(f"inline plan -> SINGLE with empty body ({t_sk:.1f}s); plain request instead")
                    plain = Section(0)
                    threading.Thread(target=run_section, args=(req, plain, 1, 1, [], max_tokens * 4, ""), daemon=True).start()
                    self._relay_sections(rid, model, [plain], t0, t_sk, strip_first_line=False)
                return
            n = len(titles)
            log(f"inline sections n={n} slots={slots} plan {t_sk:.1f}s: {titles}")
            secs = [first] + [Section(i) for i in range(1, n)]
            for i in range(1, n):
                threading.Thread(target=run_section, args=(req, secs[i], i + 1, n, titles, max_tokens), daemon=True).start()
            self._relay_sections(rid, model, secs, t0, t_sk, strip_first_line=True)
            return
        try:
            titles, raw = get_skeleton(req, min(slots, SECTIONS_MAX))
        except Exception as e:
            log(f"skeleton failed: {e}"); return self._passthrough("POST", body)
        t_sk = time.time() - t0
        if len(titles) < 3:
            log(f"skeleton -> SINGLE ({t_sk:.1f}s): {raw[:80]!r}"); return self._passthrough("POST", body)
        titles = titles[:slots]
        n = len(titles)
        log(f"sections n={n} slots={slots} skeleton {t_sk:.1f}s: {titles}")
        secs = [Section(i) for i in range(n)]
        for i in range(n):
            threading.Thread(target=run_section, args=(req, secs[i], i + 1, n, titles, max_tokens), daemon=True).start()
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
        self._sse(rid, model, "")
        self._relay_sections(rid, model, secs, t0, t_sk, strip_first_line=False)

    def _relay_sections(self, rid, model, secs, t0, t_sk, strip_first_line):
        """Stream section 0 live, then each further section in order (already generated or still streaming)."""
        total = 0
        try:
            for i, sec in enumerate(secs):
                sent = 0; skipping = strip_first_line and i == 0
                if i > 0: self._sse(rid, model, "\n\n")
                while True:
                    with sec.cv:
                        while sent >= len(sec.parts) and not sec.done: sec.cv.wait(0.5)
                        chunk = "".join(sec.parts[sent:]); sent = len(sec.parts); done = sec.done
                    if skipping:
                        # drop everything up to and including the PLAN line
                        joined = "".join(sec.parts[:sent])
                        if "\n" in joined:
                            chunk = joined.split("\n", 1)[1]; skipping = False
                        else:
                            chunk = ""
                    if chunk:
                        # a section must contain exactly one "## " heading: if a second one appears, stop relaying this section
                        emitted = getattr(sec, "emitted", "")
                        cand = emitted + chunk
                        idx = [m.start() for m in re.finditer(r"(?m)^## ", cand)]
                        if len(idx) >= 2:
                            chunk = cand[len(emitted):idx[1]].rstrip(); sec.emitted = cand[:idx[1]]
                            if chunk: self._sse(rid, model, chunk)
                            sec.cut = True; break
                        sec.emitted = cand
                        self._sse(rid, model, chunk.lstrip("\n") if not skipping and i == 0 and sent <= 2 else chunk)
                    if done and sent >= len(sec.parts): break
                total += sec.tokens
                if sec.err: self._sse(rid, model, f"\n\n[section {i+1} failed: {sec.err}]")
            el = time.time() - t0
            self._sse(rid, model, None, "stop", {"usage": {"completion_tokens": total, "prompt_tokens": 0, "total_tokens": total}})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            log(f"done n={len(secs)} tokens={total} wall={el:.1f}s -> {total/max(el,0.01):.1f} tok/s effective (plan {t_sk:.1f}s)")
        except Exception as e:
            log(f"stream error: {e}")

class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True

if __name__ == "__main__":
    log(f"start port {PORT} -> {UP} sections_max={SECTIONS_MAX} disable={DISABLE}")
    S(("127.0.0.1", PORT), H).serve_forever()
