#!/usr/bin/env python3
"""
rank-tokens.py SERVER_URL OUT.txt FILE [FILE...]

Builds a frequency-ranked token-id list for the FR-Spec draft head by tokenizing the given text files through a
running llama-server's /tokenize (so it is the model's own tokenizer, no reimplementation). Output: one id per
line, most frequent first. Use the owner's own material (Arabic + English + code) so the K-subset covers what the
model actually emits for him; ids never seen are appended in BPE order so any K is satisfiable.
"""
import sys, json, urllib.request, collections, os
URL, OUT, FILES = sys.argv[1].rstrip('/'), sys.argv[2], sys.argv[3:]
def tokenize(text):
    req = urllib.request.Request(URL + '/tokenize', data=json.dumps({"content": text, "add_special": False}).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())['tokens']
cnt = collections.Counter(); total = 0
for fp in FILES:
    try: txt = open(fp, encoding='utf-8', errors='ignore').read()
    except Exception as e: print(f"  skip {fp}: {e}"); continue
    for i in range(0, len(txt), 60000):           # chunk to keep requests small
        toks = tokenize(txt[i:i+60000]); cnt.update(toks); total += len(toks)
    print(f"  {os.path.basename(fp)}: {len(txt)} chars", flush=True)
n_vocab = json.loads(urllib.request.urlopen(URL + '/props', timeout=30).read()).get('n_vocab') or 248320
ranked = [t for t, _ in cnt.most_common()]
seen = set(ranked); ranked += [i for i in range(n_vocab) if i not in seen]
with open(OUT, 'w') as f: f.write('\n'.join(map(str, ranked)) + '\n')
cum = 0; marks = {}
for k, (t, c) in enumerate(cnt.most_common(), 1):
    cum += c
    for K in (8192, 16384, 32768):
        if k == K: marks[K] = cum / total
print(f"  {total:,} tokens, {len(cnt):,} distinct; coverage of the corpus by top-K:", {K: f"{v*100:.1f}%" for K, v in marks.items()})
print(f"  -> {OUT}")
