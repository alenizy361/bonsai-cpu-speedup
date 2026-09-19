import sys, os, json, time, threading, urllib.request
PORT=int(sys.argv[1]); K=int(sys.argv[2]); N=int(sys.argv[3]) if len(sys.argv)>3 else 96
PROMPTS=["اكتب فقرة قصيرة عن أهمية القراءة.","Explain what a hash table is in one paragraph.","اكتب ثلاث نصائح للنوم الجيد.","Write a Python function that reverses a linked list.","ما هي فوائد المشي اليومي؟","Describe the water cycle briefly.","اشرح الفرق بين الرام والقرص الصلب.","Give two tips for writing clean code.","لماذا السماء زرقاء؟","Summarize how vaccines work.","اكتب جملتين عن مدينة الرياض.","What is a binary search?","اذكر ثلاثة أنواع من الطاقة المتجددة.","Explain recursion simply.","ما هو الذكاء الاصطناعي باختصار؟","List three uses of Python."]
res=[None]*K
def go(i):
    b={"messages":[{"role":"user","content":PROMPTS[i%len(PROMPTS)]}],"max_tokens":N,"temperature":0,"cache_prompt":False}
    if os.environ.get("NOSPEC")=="1": b["speculative.n_max"]=0
    body=json.dumps(b).encode()
    req=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=900)); res[i]=(time.time()-t0, d.get("timings",{}), d.get("usage",{}))
t0=time.time(); th=[threading.Thread(target=go,args=(i,)) for i in range(K)]
[t.start() for t in th]; [t.join() for t in th]; wall=time.time()-t0
tot=sum((r[1].get("predicted_n") or r[2].get("completion_tokens",0)) for r in res if r)
per=[r[1].get("predicted_per_second",0) for r in res if r]
print(f"  K={K:2d}: {tot} توكن في {wall:.1f} ث → إجمالي {tot/wall:.2f} توكن/ث | لكل تيار ~{sum(per)/max(len(per),1):.2f}")
