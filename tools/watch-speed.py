#!/usr/bin/env python3
# llama-server SSE (chat or completion) on stdin -> live tokens, then the speed line
import sys, json, time
t0 = time.time(); n = 0; T = None
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    try:
        d = json.loads(line[6:])
    except Exception:
        continue
    c = d.get("content", "")
    r = ""
    if d.get("choices"):
        delta = d["choices"][0].get("delta") or {}
        c = c or delta.get("content") or ""
        r = delta.get("reasoning_content") or ""
    if r:
        if not globals().get("_in_think"):
            sys.stdout.write("\033[2m\033[36m🧠 thinking: "); globals()["_in_think"] = True
        sys.stdout.write(r); sys.stdout.flush(); n += 1; globals()["_nthink"] = globals().get("_nthink", 0) + 1
    if c:
        if globals().get("_in_think"):
            sys.stdout.write("\033[0m\n\n💬 answer: "); globals()["_in_think"] = False
        sys.stdout.write(c); sys.stdout.flush(); n += 1
    if d.get("timings"):
        T = d["timings"]
print("\n" + "─" * 44)
if T:
    dec = T.get("predicted_per_second", 0.0); pp = T.get("prompt_per_second", 0.0)
    ntok = T.get("predicted_n", n); ms = T.get("predicted_ms", 0.0) / 1000.0
    dn = T.get("draft_n"); da = T.get("draft_n_accepted")
    line = f"⏱ الفك: {dec:.2f} توكن/ث   |   البرومبت: {pp:.2f} توكن/ث   |   {ntok} توكن في {ms:.1f} ث"
    if dn:
        line += f"   |   مسودات مقبولة {da}/{dn} ({100.0*da/dn:.0f}%)"
    if globals().get("_nthink"): line += f"   |   أجزاء التفكير {globals()['_nthink']}"
    print("\033[0m" + line)
else:
    el = time.time() - t0
    print(f"⏱ {n} أجزاء في {el:.1f} ث ≈ {n/max(el,0.01):.2f}/ث (بلا timings من الخادم)")
