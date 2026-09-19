import struct, sys, urllib.request, json, os
URL=sys.argv[1]; OUT=sys.argv[2]; os.makedirs(OUT, exist_ok=True)
def rng(a,b):
    req=urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"}); return urllib.request.urlopen(req, timeout=120).read()
hdr=rng(0, 24*1024*1024-1); o=[8]
def raw(n): v=hdr[o[0]:o[0]+n]; o[0]+=n; return v
def u32(): return struct.unpack('<I',raw(4))[0]
def u64(): return struct.unpack('<Q',raw(8))[0]
def s(): n=u64(); return raw(n).decode('utf-8','replace')
def val(t):
    sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in sz: return raw(sz[t])
    if t==8: return s()
    if t==9:
        et=u32(); n=u64(); return [val(et) for _ in range(n)]
assert hdr[:4]==b'GGUF'; o[0]=4; ver=u32(); nt=u64(); nkv=u64(); align=32; KV={}
for _ in range(nkv):
    k=s(); t=u32(); v=val(t); KV[k]=(t,v)
    if k=='general.alignment': align=struct.unpack('<I',v)[0]
T=[]
for _ in range(nt):
    name=s(); nd=u32(); dims=[u64() for _ in range(nd)]; t=u32(); off=u64(); T.append((name,dims,t,off))
data_start=(o[0]+align-1)//align*align
TS={0:(4,1),1:(2,1),2:(18,32),3:(20,32),8:(34,32),10:(144,256),11:(110,256),12:(144,256),13:(176,256),14:(210,256),15:(292,256),16:(66,256),17:(74,256),18:(136,256),19:(80,256),20:(80,256),21:(110,256),22:(98,256),23:(136,256),29:(82,256),30:(20,32)}
def nbytes(dims,t):
    n=1
    for d in dims: n*=d
    bs,bl=TS[t]; return n//bl*bs
nl=struct.unpack('<I',KV['qwen35moe.block_count'][1])[0]
nx=KV.get('qwen35moe.nextn_predict_layers'); print(f"  block_count={nl} nextn_predict_layers={struct.unpack('<I',nx[1])[0] if nx else None} tensors={len(T)}")
want=[(n,d,t,off) for (n,d,t,off) in T if n.startswith(f'blk.{nl-1}.')]
tot=sum(nbytes(d,t) for n,d,t,off in want); print(f"  nextn block tensors: {len(want)}  total {tot/1e6:.0f} MB")
for n,d,t,off in want: print(f"    {nbytes(d,t)/1e6:8.1f} MB  type {t:2d}  {d}  {n}")
json.dump({"data_start":data_start,"align":align,"tensors":[(n,d,t,off,nbytes(d,t)) for n,d,t,off in want]}, open(os.path.join(OUT,"manifest.json"),"w"))
if len(sys.argv)>3 and sys.argv[3]=="fetch":
    for n,d,t,off,nb in [(n,d,t,off,nbytes(d,t)) for n,d,t,off in want]:
        p=os.path.join(OUT, n+".bin")
        if os.path.exists(p) and os.path.getsize(p)==nb: continue
        blob=rng(data_start+off, data_start+off+nb-1); open(p,'wb').write(blob); print(f"    fetched {n} {len(blob)/1e6:.1f} MB", flush=True)
    print("  fetch done")
