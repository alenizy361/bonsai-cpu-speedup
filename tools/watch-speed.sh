#!/bin/bash
# watch-speed.sh [prompt] — بثّ حي عبر واجهة الدردشة لخدمة Bonsai (:4791) مع رقم السرعة في النهاية
P="${1:-اكتب فقرة من مئتي كلمة عن كيف ستبدو مدينة الرياض عام 2040، بأسلوب أدبي وبتفاصيل ملموسة.}"
N="${N:-260}"; TEMP="${TEMP:-0.7}"
echo "▶ البرومبت: $P     (T=$TEMP, thinking=${THINK:-0})"; echo "────────────────────────────────────────────"
BODY=$(python3 -c 'import json,sys; b={"messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":float(sys.argv[3]),"stream":True,"cache_prompt":False}
if sys.argv[4]=="1": b["chat_template_kwargs"]={"enable_thinking":True}
print(json.dumps(b))' "$P" "$N" "$TEMP" "${THINK:-0}")
curl -sN http://127.0.0.1:4791/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY" | python3 -u /home/abdulaziz/bonsai/tools/watch-speed.py
echo
if [ -t 0 ] && [ -z "${NO_LOOP:-}" ]; then read -rp "اضغط Enter لتجربة أخرى، أو Ctrl+C للخروج... " && exec "$0"; fi
