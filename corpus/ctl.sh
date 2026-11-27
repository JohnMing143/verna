#!/bin/bash
# Verna 语料库收集器：systemd 托管（单元 srh-corpus），低优先级，做完一轮自动退出；可重复 start，已完成的文件会跳过
ROOT=/root/shuorenhua
UNIT=srh-corpus
case "$1" in
  start)
    if systemctl is-active --quiet $UNIT; then echo "已经在跑"; exit 0; fi
    systemctl reset-failed $UNIT 2>/dev/null
    systemd-run --unit=$UNIT --description="Verna 语料库收集（shuorenhua）" \
      -p Nice=15 -p IOSchedulingClass=idle -p CPUQuota=80% -p MemoryMax=3G \
      --working-directory=$ROOT $ROOT/.venv/bin/python $ROOT/corpus/collect.py "${@:2}"
    ;;
  stop) systemctl stop $UNIT ;;
  status) systemctl status $UNIT --no-pager 2>/dev/null | head -4; tail -15 $ROOT/corpus/collect.log ;;
  log) tail -f $ROOT/corpus/collect.log ;;
  *) echo "用法: $0 start|stop|status|log [collect.py 参数]"; exit 1 ;;
esac
