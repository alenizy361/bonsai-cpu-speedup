#!/usr/bin/env python3
"""graft_moe_mtp.py IN.gguf BLOBDIR OUT.gguf — append a nextn (MTP) block fetched as raw tensor blobs (manifest.json) to a
qwen35moe GGUF: adds the blk.<n_layer>.* tensors, bumps block_count by 1 and sets nextn_predict_layers=1. Data copied byte for byte."""
import struct, sys, os, json
IN, BLOBS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
man = json.load(open(os.path.join(BLOBS, "manifest.json")))
class Rd:
    def __init__(s, b): s.b=b; s.o=0
    def raw(s,n): v=s.b[s.o:s.o+n]; s.o+=n; return v
    def u32(s): return struct.unpack("<I", s.raw(4))[0]
    def u64(s): return struct.unpack("<Q", s.raw(8))[0]
    def str(s): n=s.u64(); return s.raw(n).decode("utf-8","replace")
    def val(s,t):
        sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t in sz: return s.raw(sz[t])
        if t==8: return s.str()
        if t==9:
            et=s.u32(); n=s.u64(); return (et,[s.val(et) for _ in range(n)])
        raise ValueError(t)
def parse(buf):
    r=Rd(buf); assert r.raw(4)==b"GGUF"; ver=r.u32(); nt=r.u64(); nkv=r.u64(); kv=[]
    for _ in range(nkv): k=r.str(); t=r.u32(); v=r.val(t); kv.append([k,t,v])
    tl=[]
    for _ in range(nt): name=r.str(); nd=r.u32(); dims=[r.u64() for _ in range(nd)]; t=r.u32(); off=r.u64(); tl.append([name,dims,t,off])
    return ver,kv,tl,r.o
def wstr(f,s): b=s.encode(); f.write(struct.pack("<Q",len(b))); f.write(b)
def wval(f,t,v):
    if t==8: wstr(f,v); return
    if t==9:
        et,vals=v; f.write(struct.pack("<I",et)); f.write(struct.pack("<Q",len(vals)))
        for x in vals: wval(f,et,x)
        return
    f.write(v)
def align_up(x,a): return (x+a-1)//a*a
head=open(IN,"rb").read(64*1024*1024); ver,kv,tl,hdr_end=parse(head)
ALIGN=next((struct.unpack("<I",v)[0] for k,t,v in kv if k=="general.alignment"),32); data_start=align_up(hdr_end,ALIGN); fsize=os.path.getsize(IN)
names={t[0] for t in tl}
arch=next(v for k,t,v in kv if k=="general.architecture"); assert arch=="qwen35moe", arch
nl=next(struct.unpack("<I",v)[0] for k,t,v in kv if k==f"{arch}.block_count"); print(f"[1/3] {IN}: block_count={nl}, tensors={len(tl)}")
assert not any(n.startswith(f"blk.{nl}.") for n in names), "already has a nextn block"
# kv edits
kv2=[]; seen_nextn=False
for k,t,v in kv:
    if k==f"{arch}.block_count": v=struct.pack("<I",nl+1)
    if k==f"{arch}.nextn_predict_layers": v=struct.pack("<I",1); seen_nextn=True
    kv2.append([k,t,v])
if not seen_nextn: kv2.append([f"{arch}.nextn_predict_layers",4,struct.pack("<I",1)])
# new tensors appended after existing data
orig_bytes=fsize-data_start; cur=align_up(orig_bytes,ALIGN); new_tl=[t for t in tl]; blobs=[]
for name,dims,typ,srcoff,nb in man["tensors"]:
    p=os.path.join(BLOBS,name+".bin"); assert os.path.getsize(p)==nb, name
    new_tl.append([name,dims,typ,cur]); blobs.append((p,cur,nb)); cur=align_up(cur+nb,ALIGN)
print(f"[2/3] appending {len(blobs)} tensors ({sum(b[2] for b in blobs)/1e6:.0f} MB) as blk.{nl}.*; nextn_predict_layers=1, block_count={nl+1}")
with open(OUT,"wb") as o:
    o.write(b"GGUF"); o.write(struct.pack("<I",ver)); o.write(struct.pack("<Q",len(new_tl))); o.write(struct.pack("<Q",len(kv2)))
    for k,t,v in kv2: wstr(o,k); o.write(struct.pack("<I",t)); wval(o,t,v)
    for name,dims,t,off in new_tl:
        wstr(o,name); o.write(struct.pack("<I",len(dims)))
        for d in dims: o.write(struct.pack("<Q",d))
        o.write(struct.pack("<I",t)); o.write(struct.pack("<Q",off))
    pad=align_up(o.tell(),ALIGN)-o.tell(); o.write(b"\0"*pad); nds=o.tell()
    with open(IN,"rb") as f:
        f.seek(data_start); left=orig_bytes
        while left:
            c=f.read(min(64<<20,left)); o.write(c); left-=len(c)
    pos=orig_bytes
    for p,off,nb in blobs:
        o.write(b"\0"*(off-pos)); o.write(open(p,"rb").read()); pos=off+nb
print(f"[3/3] wrote {OUT}: {os.path.getsize(OUT)/1e9:.3f} GB")
