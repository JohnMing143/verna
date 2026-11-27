#!/usr/bin/env python3
"""测试用的假 agy acp：从 stdin 读换行分隔的 JSON-RPC，往 stdout 回。

行为由环境变量控制：
  FAKE_AGY_MODE   normal（默认）/ crash_on_prompt / warn / quota / slow / auth_text / auth_code
  FAKE_AGY_PIDFILE  启动时把 pid 追加写进去（测试数进程数用）
  FAKE_AGY_LOG      每条请求、收到的 cancel、启动参数都记一行 JSON，方便断言
"""
import json
import os
import re
import sys
import threading
import time

MODE = os.environ.get("FAKE_AGY_MODE", "") or "normal"
PIDFILE = os.environ.get("FAKE_AGY_PIDFILE", "")
LOG = os.environ.get("FAKE_AGY_LOG", "")

OUT_LOCK = threading.Lock()
SESSIONS = {}
NEXT_SID = [0]

# 模型名故意有 value 和 id 两种写法，客户端两种都要认
MODELS = ["gemini-4-neon", "gemini-4-carbon", "gemini-4-argon", {"id": "gemini-3.8-flash"}, "claude-opus-5-5"]
LEVELS = ["low", "medium", "high", "xhigh"]
PRETTY = {"gemini-4-neon": "Gemini 4 Neon", "gemini-4-carbon": "Gemini 4 Carbon",
          "gemini-4-argon": "Gemini 4 Argon", "gemini-3.8-flash": "Gemini 3.8 Flash",
          "claude-opus-5-5": "Claude Opus 5.5"}


def log(rec):
    if not LOG:
        return
    with OUT_LOCK:
        with open(LOG, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def send(msg):
    with OUT_LOCK:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def model_names():
    out = []
    for m in MODELS:
        out.append(m if isinstance(m, str) else m.get("value") or m.get("id"))
    return out


def configs(sess):
    lv = LEVELS + (["max"] if sess["model"].startswith("claude") else [])
    return [{"id": "model", "currentValue": sess["model"],
             "options": [dict(m) if isinstance(m, dict) else {"value": m} for m in MODELS]},
            {"id": "thought_level", "currentValue": sess["level"],
             "options": [{"value": x} for x in lv]}]


def error(mid, code, message):
    send({"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}})


def chunk(sid, text):
    send({"jsonrpc": "2.0", "method": "session/update",
          "params": {"sessionId": sid,
                     "update": {"sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": text}}}})


def usage():
    return {"inputTokens": 9435, "outputTokens": 7, "cachedReadTokens": 3,
            "cachedWriteTokens": 0, "thoughtTokens": 5, "totalTokens": 9450}


def answer_thread(mid, sid, sess, text):
    marker = (re.search(r"MARK-[\w]+", text) or [text[:20]])[0]
    parts = [f"[{marker}] model={sess['model']} level={sess['level']} part-1",
             " part-2", " part-3"]
    if MODE == "warn":
        parts = ["\n⚠ failed to construct executor: boom ", "part-2"]
    sess["pending"].add(mid)
    for i, part in enumerate(parts):
        if sess["cancel"].is_set():
            send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "cancelled"}})
            sess["pending"].discard(mid)
            return
        # 夹一个会被忽略的通知，客户端不该把它当正文
        send({"jsonrpc": "2.0", "method": "session/update",
              "params": {"sessionId": sid, "update": {"sessionUpdate": "usage_update"}}})
        chunk(sid, part)
        time.sleep(0.03)
    sess["pending"].discard(mid)
    send({"jsonrpc": "2.0", "id": mid,
          "result": {"stopReason": "end_turn", "usage": usage()}})


def slow_thread(mid, sid, sess):
    sess["pending"].add(mid)
    sess["cancel"].wait(30)
    sess["pending"].discard(mid)
    if sess["cancel"].is_set():
        send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "cancelled"}})
    else:
        chunk(sid, "slow done")
        send({"jsonrpc": "2.0", "id": mid,
              "result": {"stopReason": "end_turn", "usage": usage()}})


def main():
    if PIDFILE:
        with open(PIDFILE, "a") as f:
            f.write(f"{os.getpid()}\n")
            f.flush()
    log({"argv": sys.argv[1:]})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}

        if method == "session/cancel":
            sid = params.get("sessionId")
            sess = SESSIONS.get(sid)
            log({"method": method, "params": params})
            if sess:
                sess["cancel"].set()  # 等 prompt 线程看到后自己回 cancelled
            continue

        log({"method": method, "params": params})

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid,
                  "result": {"protocolVersion": 1, "agentCapabilities": {}}})
        elif method == "session/new":
            if MODE == "auth_text":
                error(mid, -32000, "Authentication required")
                continue
            if MODE == "auth_code":
                error(mid, -32000, "login expired, please sign in again")
                continue
            NEXT_SID[0] += 1
            sid = f"sess-{NEXT_SID[0]}"
            SESSIONS[sid] = {"model": "gemini-4-neon", "level": "high",
                             "cancel": threading.Event(), "pending": set()}
            send({"jsonrpc": "2.0", "id": mid,
                  "result": {"sessionId": sid, "configOptions": configs(SESSIONS[sid])}})
        elif method == "session/set_config_option":
            sess = SESSIONS.get(params.get("sessionId"))
            cid, value = params.get("configId"), params.get("value")
            if sess is None:
                error(mid, -32602, f"Unknown session: {params.get('sessionId')}")
            elif cid == "model":
                if value not in model_names():
                    error(mid, -32602, f"Unknown model ID: {value}")
                    continue
                sess["model"] = value
                send({"jsonrpc": "2.0", "id": mid, "result": {"configOptions": configs(sess)}})
            elif cid == "thought_level":
                lv = LEVELS + (["max"] if sess["model"].startswith("claude") else [])
                if value not in lv:
                    error(mid, -32602,
                          f'Thinking level "{value}" is not offered for model '
                          f'"{PRETTY.get(sess["model"], sess["model"])}"')
                    continue
                sess["level"] = value
                send({"jsonrpc": "2.0", "id": mid, "result": {"configOptions": configs(sess)}})
            else:
                error(mid, -32602, f"Unknown config ID: {cid}")
        elif method == "session/prompt":
            sess = SESSIONS.get(params.get("sessionId"))
            if sess is None:
                error(mid, -32602, f"Unknown session: {params.get('sessionId')}")
                continue
            text = "".join(p.get("text", "") for p in params.get("prompt") or [] if p.get("type") == "text")
            if MODE == "crash_on_prompt":
                sys.stdout.flush()
                os._exit(3)
            if MODE == "quota":
                error(mid, -32010, "RESOURCE_EXHAUSTED: daily quota used up")
                continue
            if MODE == "slow":
                threading.Thread(target=slow_thread, args=(mid, params.get("sessionId"), sess),
                                 daemon=True).start()
                continue
            threading.Thread(target=answer_thread, args=(mid, params.get("sessionId"), sess, text),
                             daemon=True).start()
        elif method == "session/close":
            SESSIONS.pop(params.get("sessionId"), None)
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif mid is not None:
            error(mid, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
