#!/usr/bin/env bash
# 管线的启动、停止和查看。用 systemd 托管：崩溃自动重启，会话断开也不受影响。
#   bash autopilot/ctl.sh start | stop | status | log | pause | resume
set -euo pipefail
ROOT=/root/shuorenhua
UNIT=srh-autopilot
CTL=$ROOT/data/v3/autopilot/control.json
case "${1:-status}" in
  start)
    mkdir -p "$ROOT/data/v3/autopilot"
    systemctl reset-failed $UNIT 2>/dev/null || true   # stop 超时被强杀后单元会留在 failed 状态，不清掉起不来
    systemd-run --unit=$UNIT --description="Verna 数据管线（shuorenhua）" \
      --property=Restart=on-failure --property=RestartSec=120 \
      --property=MemoryAccounting=yes --property=MemoryHigh=6G --property=MemoryMax=7G --property=MemorySwapMax=512M \
      --property=OOMScoreAdjust=500 \
      --working-directory=$ROOT \
      --property=EnvironmentFile=-$ROOT/autopilot/env \
      --setenv=PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin --setenv=HOME=/root --setenv=PYTHONUNBUFFERED=1 \
      $ROOT/.venv/bin/python $ROOT/autopilot/run.py
    sleep 3; systemctl status $UNIT --no-pager | head -6 ;;
  stop)    systemctl stop $UNIT || true; systemctl reset-failed $UNIT 2>/dev/null || true ;;
  status)  systemctl status $UNIT --no-pager | head -6; python3 -c "import json; s=json.load(open('$ROOT/data/v3/autopilot/status.json')); print(s.get('state'), s.get('updated')); print('额度', s.get('quota'))" ;;
  log)     tail -n "${2:-40}" $ROOT/data/v3/autopilot/run.log ;;
  pause)   python3 -c "import json,os; p='$CTL'; c=json.load(open(p)) if os.path.exists(p) else {}; c['pause']=True; json.dump(c,open(p,'w'),ensure_ascii=False)"; echo 已暂停（当前环节做完后生效） ;;
  resume)  python3 -c "import json,os; p='$CTL'; c=json.load(open(p)) if os.path.exists(p) else {}; c['pause']=False; json.dump(c,open(p,'w'),ensure_ascii=False)"; echo 已恢复 ;;
  *) echo "用法：$0 start|stop|status|log [行数]|pause|resume" ;;
esac
