import json,sys,re,time,urllib.request
tag=sys.argv[1]
Q=[("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost? Reply with the number only.","5"),
("I have 3 boxes with 4, 5 and 6 apples. I eat 2 apples from each box. How many apples remain in total? Reply with the number only.","9"),
("What is 17 × 24? Reply with the number only.","408"),
("A train travels 180 km in 2.5 hours. What is its average speed in km/h? Number only.","72"),
("كم عدد الأيام في السنة الكبيسة؟ اكتب الرقم فقط.","366"),
("ما عاصمة أستراليا؟ اكتب اسم المدينة فقط.","كانبرا|كانبيرا|canberra"),
("من مؤلف كتاب «المقدمة» الشهير في علم العمران؟ اكتب الاسم فقط.","ابن خلدون|khaldun"),
("In Python, what does len(set([1,2,2,3,3,3])) return? Reply with the number only.","3"),
("What is the smallest prime number greater than 31? Number only.","37"),
("In strict JSON, is a trailing comma after the last array element valid? Answer with yes or no only.","no"),
("كم يساوي 15% من 240؟ اكتب الرقم فقط.","36"),
("Sort these words alphabetically: pear, apple, fig. Reply with them comma-separated, nothing else.","apple, fig, pear"),
("A rectangle is 7 cm by 12 cm. What is its perimeter in cm? Number only.","38"),
("إذا كان اليوم الثلاثاء، فما اليوم بعد 10 أيام؟ اكتب اسم اليوم فقط.","الجمعة|friday"),
("Which is larger: 9.11 or 9.9? Reply with the larger number only.","9.9"),
("How many letters 'r' are in the word 'strawberry'? Number only.","3")]
score=0; t0=time.time(); ntok=0; tsec=0
for q,a in Q:
    body={"messages":[{"role":"user","content":q}],"max_tokens":24,"temperature":0}
    req=urllib.request.Request("http://127.0.0.1:4791/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req,timeout=600)); c=d["choices"][0]["message"]["content"].strip(); t=d.get("timings",{})
    ntok+=t.get("predicted_n",0); tsec+=t.get("predicted_ms",0)/1000
    norm=re.sub(r"[*_`\"'!؟?]+","",c.lower()).replace("،",",").strip()
    ok=any(x.lower() in norm for x in a.split("|")); score+=ok
    print(f"  [{tag}] {'✓' if ok else '✗'} {c[:40]!r:44s} expected {a}")
print(f"  [{tag}] SCORE {score}/{len(Q)}  | decode {ntok/max(tsec,0.01):.2f} tok/s over {ntok} tokens")
