#!/bin/bash
# model-switch.sh moe|bonsai — swaps the live model behind :4791 (Open WebUI keeps the same model id "bonsai-27b-uncensored")
D=/home/abdulaziz/.config/systemd/user/bonsai-llama.service.d; BIN=/home/abdulaziz/bonsai/bin/llama-server-mtp
case "$1" in
  moe-mtp)   # MoE + grafted MTP head — measured SLOWER (verify+draft cost ~2x a plain token on this MoE)
    cat > $D/99-model.conf <<EOC
[Service]
ExecStart=
ExecStart=$BIN -m /data/models/qwen36-35b-a3b-ablit-IQ2_M-mtp.gguf --alias bonsai-27b-uncensored -t 6 -tb 12 -c 16384 -ctk q8_0 -ctv q8_0 --flash-attn on --parallel 2 --cache-ram 0 --spec-type draft-mtp --spec-draft-n-max 2 --jinja --reasoning-format deepseek --chat-template-kwargs '{"enable_thinking":false}' --host 127.0.0.1 --port 4791
EOC
    ;;
  moe)   # default: MoE without speculation (fastest single stream here)
    cat > $D/99-model.conf <<EOC
[Service]
ExecStart=
ExecStart=$BIN -m /data/models/qwen36-35b-a3b-ablit-IQ2_M.gguf --alias bonsai-27b-uncensored -t 6 -tb 12 -c 65536 -ctk q8_0 -ctv q8_0 --flash-attn on --parallel 2 --cache-ram 0 --jinja --reasoning-format deepseek --chat-template-kwargs '{"enable_thinking":false}' --host 127.0.0.1 --port 4791
EOC
    ;;
  bonsai) rm -f $D/99-model.conf ;;
  *) echo "usage: $0 moe|moe-nospec|bonsai"; exit 1 ;;
esac
systemctl --user daemon-reload; systemctl --user restart bonsai-llama
for i in $(seq 1 120); do curl -s --max-time 2 127.0.0.1:4791/health | grep -q '"ok"' && break; sleep 2; done
echo "الخدمة: $(systemctl --user is-active bonsai-llama) — النموذج الآن: $1"
