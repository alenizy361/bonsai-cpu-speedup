#!/usr/bin/env python3
"""
make-frspec-head.py IN.gguf OUT.gguf K [ranking.txt]

FR-Spec-style restricted draft head for the grafted MTP block (2026-09-18). The MTP draft graph multiplies by the
FULL output.weight [5120 x 248320] (338 MB, 1.27 G MAC) on every draft step. This writes two extra tensors:

  blk.64.nextn.shared_head_head  PQ2_0 [5120, K]   rows of output.weight for the K chosen token ids (byte-exact copies)
  d2t                            I64   [K]         draft row -> target token id

graph_mtp already prefers nextn.shared_head_head when present; with d2t the graph scatters the K logits into a
-inf-filled full-vocab vector (the EAGLE-3 / DFlash pattern), so the draft can only propose tokens in the subset
while the target still verifies with the full model -> the output distribution is unchanged (exact).

Ranking: ranking.txt = one token id per line, most frequent first (build it from the owner's own text via the
server's /tokenize). Without it, ids 0..K-1 are used (BPE merge order is a rough frequency proxy).
"""
import struct, sys, os

IN, OUT, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
RANK = sys.argv[4] if len(sys.argv) > 4 else None
GGML_TYPE_PQ2_0, GGML_TYPE_I64 = 142, 27
QK_PQ2_0, BLOCK_BYTES = 128, 34

class Rd:
    def __init__(s, b): s.b = b; s.o = 0
    def raw(s, n):
        if s.o + n > len(s.b): raise EOFError("header buffer too small")
        v = s.b[s.o:s.o+n]; s.o += n; return v
    def u32(s): return struct.unpack("<I", s.raw(4))[0]
    def u64(s): return struct.unpack("<Q", s.raw(8))[0]
    def str(s): n = s.u64(); return s.raw(n).decode("utf-8", "replace")
    def val(s, t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t in sz: return s.raw(sz[t])
        if t == 8: return s.str()
        if t == 9:
            et = s.u32(); n = s.u64(); return (et, [s.val(et) for _ in range(n)])
        raise ValueError(f"unknown gguf value type {t}")

def parse(buf):
    r = Rd(buf)
    assert r.raw(4) == b"GGUF", "not a GGUF"
    ver = r.u32(); nt = r.u64(); nkv = r.u64()
    kv = []
    for _ in range(nkv):
        k = r.str(); t = r.u32(); v = r.val(t); kv.append((k, t, v))
    tl = []
    for _ in range(nt):
        name = r.str(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]; t = r.u32(); off = r.u64()
        tl.append([name, dims, t, off])
    return ver, kv, tl, r.o

def wstr(f, s): b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)
def wval(f, t, v):
    if t == 8: wstr(f, v); return
    if t == 9:
        et, vals = v; f.write(struct.pack("<I", et)); f.write(struct.pack("<Q", len(vals)))
        for x in vals: wval(f, et, x)
        return
    f.write(v)  # raw bytes for scalar types (kept verbatim)
def align_up(x, a): return (x + a - 1) // a * a

with open(IN, "rb") as f: head = f.read(64 * 1024 * 1024)
ver, kv, tl, hdr_end = parse(head)
ALIGN = next((struct.unpack("<I", v)[0] for k, t, v in kv if k == "general.alignment"), 32)
data_start = align_up(hdr_end, ALIGN)
fsize = os.path.getsize(IN)
print(f"[1/4] {IN}: tensors={len(tl)} kv={len(kv)} data_start={data_start} size={fsize/1e9:.3f} GB", flush=True)

names = {t[0]: t for t in tl}
assert "output.weight" in names, "no output.weight"
assert "blk.64.nextn.eh_proj.weight" in names, "no MTP block (blk.64.*) — graft the MTP head first"
assert "blk.64.nextn.shared_head_head.weight" not in names and "d2t" not in names, "already has a restricted head"
ow = names["output.weight"]; assert ow[2] == GGML_TYPE_PQ2_0, f"output.weight type {ow[2]} != PQ2_0"
n_embd, n_vocab = ow[1][0], ow[1][1]
assert n_embd % QK_PQ2_0 == 0
row_bytes = n_embd // QK_PQ2_0 * BLOCK_BYTES

# token ranking
if RANK:
    ids = []
    seen = set()
    for line in open(RANK):
        line = line.strip()
        if not line: continue
        i = int(line.split()[0])
        if 0 <= i < n_vocab and i not in seen: seen.add(i); ids.append(i)
        if len(ids) >= K: break
    assert len(ids) == K, f"ranking has only {len(ids)} usable ids (< K={K})"
    src = f"ranking file {RANK}"
else:
    ids = list(range(K)); src = "BPE id order (proxy)"
print(f"[2/4] head rows: K={K} from {src}; row={row_bytes} B -> {K*row_bytes/1e6:.1f} MB", flush=True)

# read the K rows of output.weight (byte-exact)
with open(IN, "rb") as f:
    f.seek(data_start + ow[3]); ow_bytes = f.read(n_vocab * row_bytes)
head_data = b"".join(ow_bytes[i*row_bytes:(i+1)*row_bytes] for i in ids)
d2t_data = struct.pack("<%dq" % K, *ids)
del ow_bytes

# new tensor infos appended after existing data
orig_data_bytes = fsize - data_start
off_head = align_up(orig_data_bytes, ALIGN)
off_d2t  = align_up(off_head + len(head_data), ALIGN)
new_tl = [t for t in tl] + [
    ["blk.64.nextn.shared_head_head.weight", [n_embd, K], GGML_TYPE_PQ2_0, off_head],
    ["d2t", [K], GGML_TYPE_I64, off_d2t],
]
print(f"[3/4] writing {OUT}", flush=True)
# The rows are copies of output.weight, which is Hadamard-ROTATED: the graph rotates the mul_mat input for every tensor
# listed in prism.hadamard.weight_names (sign data is keyed by input width, 5120 here, so no per-name sign entry is
# needed). Register the new head under that list or its logits are garbage (measured: 0% acceptance without this).
HEAD_NAME = "blk.64.nextn.shared_head_head.weight"
kv_out = []
registered = False
for k, t, v in kv:
    if k == "prism.hadamard.weight_names" and t == 9:
        et, vals = v
        assert et == 8 and "output.weight" in vals, "unexpected prism.hadamard.weight_names"
        if HEAD_NAME not in vals: vals = vals + [HEAD_NAME]
        v = (et, vals); registered = True
    kv_out.append((k, t, v))
assert registered, "prism.hadamard.weight_names not found — cannot register the rotated head"
print(f"      registered {HEAD_NAME} in prism.hadamard.weight_names (HADAMARD-ROTATED head)", flush=True)
kv = kv_out
with open(OUT, "wb") as o:
    o.write(b"GGUF"); o.write(struct.pack("<I", ver)); o.write(struct.pack("<Q", len(new_tl))); o.write(struct.pack("<Q", len(kv)))
    for k, t, v in kv: wstr(o, k); o.write(struct.pack("<I", t)); wval(o, t, v)
    for name, dims, t, off in new_tl:
        wstr(o, name); o.write(struct.pack("<I", len(dims)))
        for d in dims: o.write(struct.pack("<Q", d))
        o.write(struct.pack("<I", t)); o.write(struct.pack("<Q", off))
    pad = align_up(o.tell(), ALIGN) - o.tell(); o.write(b"\0" * pad)
    new_data_start = o.tell()
    with open(IN, "rb") as f:
        f.seek(data_start); left = orig_data_bytes; CH = 64 << 20
        while left:
            c = f.read(min(CH, left));
            if not c: raise IOError("short read")
            o.write(c); left -= len(c)
    o.write(b"\0" * (off_head - orig_data_bytes)); o.write(head_data)
    o.write(b"\0" * (off_d2t - (off_head + len(head_data)))); o.write(d2t_data)
    total = o.tell()
print(f"[4/4] done: {total/1e9:.3f} GB, data_start {data_start} -> {new_data_start}, +{len(head_data)/1e6:.1f} MB head, +{len(d2t_data)/1e3:.0f} KB d2t", flush=True)
# verify
with open(OUT, "rb") as f: vh = f.read(64 * 1024 * 1024)
_, vkv, vtl, _ = parse(vh)
vn = {t[0]: t for t in vtl}
ok = ("blk.64.nextn.shared_head_head.weight" in vn and vn["blk.64.nextn.shared_head_head.weight"][1] == [n_embd, K]
      and "d2t" in vn and vn["d2t"][1] == [K] and len(vtl) == len(tl) + 2)
print("      verify:", "OK" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
