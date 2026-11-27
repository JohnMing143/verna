# Verna 评委站后端

`server.py` 是投票接口，`admin.py` 是题库和导出命令，`test_server.py` 是测试。只用 Python 3.11 标准库（`http.server` + `sqlite3`），不需要安装任何包。

目录里的其他文件：

- `verna-vote.service`：systemd 单元样板（安装由用户完成）
- `nginx-api.conf.snippet`：nginx 的 `location /api/` 样板

## 启动

```sh
cd /root/shuorenhua/vote/server
python3 server.py --db /var/lib/verna/verna.db --port 8790
```

- 只监听 `127.0.0.1`，前面由 nginx 反代。
- 第一次启动会自动建库、建表，并开 WAL。数据库目录不存在时会创建。
- IP 盐按这个顺序取：
  1. 环境变量 `VERNA_IP_SALT`；
  2. 数据库同目录的 `ip_salt` 文件（不存在就生成，权限 600，之后复用）。
- 可选 `--verbose` 打印每个请求到标准错误。不加时只打印启动信息和异常堆栈。
- 数据库里只存 `sha256(盐 + IP)` 的前 16 位十六进制，不存原始 IP。

测试：

```sh
cd /root/shuorenhua/vote
python -m unittest vote/server/test_server.py
```

测试会自己在临时目录建库、用随机端口起服务，测完关闭并删除临时目录。

## 接口

所有接口挂在 `/api/` 下（nginx 保留前缀）。直接访问 `127.0.0.1:8790` 时带不带 `/api/` 都可以。

请求体只接受 `Content-Type: application/json`，上限 16 KB（16384 字节），超过返回 413。响应一律是 JSON：

- 成功：`{"ok": true, ...}`
- 失败：`{"ok": false, "error": "<代码>"}`

没有 CORS 头，前端必须和接口同源。

### 错误代码

| 代码 | 状态码 | 含义 |
| --- | --- | --- |
| `not_found` | 404 | 路径不存在 |
| `method_not_allowed` | 405 | 方法不对（响应带 `Allow`） |
| `bad_content_type` | 415 | 不是 `application/json` |
| `bad_length` | 400 | 缺少/非法的 `Content-Length` |
| `too_large` | 413 | 请求体超过 16 KB |
| `bad_json` | 400 | 请求体不是 JSON 对象 |
| `bad_judge` | 400 | 评委编号不符合 `^j[a-z0-9]{8,40}$` |
| `bad_item` | 400 | 题号缺失或不是字符串 |
| `item_not_found` | 404 | 题号不存在 |
| `item_inactive` | 400 | 题目已停用（status != active） |
| `bad_choice` | 400 | choice 不是 a/b/src |
| `bad_left` | 400 | left 不是 a/b |
| `bad_ms` | 400 | ms 不是 0–86400000 的整数 |
| `bad_reasons` | 400 | reasons 不是数组、超过 12 条或单条超过 40 字 |
| `bad_note` | 400 | note 不是字符串或超过 300 字 |
| `consent_required` | 400 | 投稿没有勾选同意 |
| `bad_text` | 400 | 投稿正文不是字符串 |
| `too_short` | 400 | 投稿汉字数不足 50 |
| `too_long` | 400 | 投稿正文超过 3000 字 |
| `bad_model` | 400 | 模型不在允许列表里 |
| `bad_scene` | 400 | 用途超过 100 字 |
| `bad_nickname` | 400 | 署名超过 30 字 |
| `duplicate` | 409 | 这段文字已经有人投过 |
| `limit` | 429 | 当天投稿次数达到上限 |
| `internal` | 500 | 未预料的服务器错误（详见标准错误日志） |

### `GET /api/next?judge=<编号>&n=<1-10>&from=<渠道>`

派题。`n` 默认 5，超出范围会截到 1–10；`from` 可选，最长 40 字。

```json
{"ok": true, "items": [{"id": "...", "genre": "...", "source": "...", "a": "...", "b": "..."}]}
```

- 只返回 `status = active`、这个评委没投过也没跳过的题。
- 先按票数从少到多，同票数随机。
- 同一次返回里同一 genre 最多 2 道；不够 `n` 时放宽补足。
- 评委第一次出现时写入 `judges`，`from` 只在这一次记录；之后不再改。
- 返回里永远没有 `a_system`、`b_system`、`kind`、`probe_answer`、`meta`。

### `POST /api/vote`

```json
{"item": "...", "judge": "j...", "choice": "a", "left": "a", "ms": 3200,
 "reasons": ["good:读着顺"], "note": ""}
```

- `choice`：`a` / `b` / `src`（原文更好）；`left`：左边放的是 `a` 还是 `b`。
- `ms`：整数，0–86,400,000。
- `reasons`：最多 12 条、每条最多 40 字的字符串数组。
- `note`：最多 300 字。
- 同一评委对同一题重复提交是覆盖（前端"改选"和"投完补理由"都靠它），`updated` 更新。

### `POST /api/unvote`

```json
{"item": "...", "judge": "j..."}
```

删除这条票。没有票也返回成功（幂等）。

### `POST /api/skip`

```json
{"item": "...", "judge": "j..."}
```

记一次跳过。之后 `/api/next` 不再给这个评委这道题。

### `POST /api/submit`

```json
{"judge": "j...", "text": "...", "model": "", "scene": "", "nickname": "", "consent": true}
```

- `consent` 必须严格为 `true`。
- `text` 去掉首尾空白后：汉字数 ≥ 50、总长 ≤ 3000。
- `model` 只能是空字符串或：`不清楚`、`ChatGPT`、`Claude`、`Gemini`、`DeepSeek`、`豆包`、`Kimi`、`通义千问`、`文心一言`、`其他`。
- `scene` ≤ 100 字，`nickname` ≤ 30 字。
- 去重按去掉所有空白后的文本算 sha256，重复返回 409 `duplicate`。
- 同一评委每天最多 20 篇，同一 IP 哈希每天最多 60 篇（UTC 自然日），超过返回 429 `limit`。
- 存为 `pending`，不会进题库。

成功返回 `{"ok": true, "id": 1, "status": "pending"}`。

### `GET /api/stats`

```json
{"ok": true, "judges": 0, "votes": 0, "items": 0, "submissions": 0}
```

四个表的总行数（`items` 包含已停用的题）。

## 管理命令

`--db` 放在子命令前后都可以。

```sh
cd /root/shuorenhua/vote/server

# 导入题库：每行一个 JSON，已存在的 id 跳过
python3 admin.py --db /var/lib/verna/verna.db import-items items.jsonl

# 停用题目（status 改为 retired，票和跳过记录保留）
python3 admin.py --db /var/lib/verna/verna.db retire demo1 demo2

# 导出全部投票（JSONL 到标准输出）
python3 admin.py --db /var/lib/verna/verna.db export-votes > votes.jsonl

# 只导出待审投稿；不传 --status 就导出全部
python3 admin.py --db /var/lib/verna/verna.db export-submissions --status pending

# 各表计数（JSON）
python3 admin.py --db /var/lib/verna/verna.db stats
```

`import-items` 的每行格式：

```json
{"id": "demo1", "genre": "职场", "source": "原文……", "a": "改写 A……", "b": "改写 B……",
 "a_system": "system-a", "b_system": "system-b", "kind": "normal", "probe_answer": null,
 "meta": {"source": "..."}}
```

必填字符串：`id`、`genre`、`source`、`a`、`b`、`a_system`、`b_system`。`kind` 只能是 `normal` / `probe`，`probe_answer` 只能是 `a` / `b` / `null`。`meta` 可以是对象或字符串。导入不覆盖已有 id（要更新就自己操作数据库或换新 id）。

`export-votes` 输出字段：`item_id, judge_id, choice, left_side, ms, reasons`（数组）、`note, ip_hash, created, updated`。

`export-submissions` 输出字段：`id, judge_id, text, model, scene, nickname, ip_hash, text_hash, status, created`。

## 部署

1. 建目录和单元（数据库目录由 `StateDirectory=verna` 自动创建，属主 root）：

   ```sh
   install -d -m 700 /var/lib/verna
   cp /root/shuorenhua/vote/server/verna-vote.service /etc/systemd/system/
   systemctl daemon-reload
   systemctl enable --now verna-vote
   systemctl status verna-vote
   ```

   单元按任务书用 `User=root`、`Restart=always`，命令是
   `/usr/bin/python3 /root/shuorenhua/vote/server/server.py --db /var/lib/verna/verna.db --port 8790`。

2. 导入第一批题库：

   ```sh
   cd /root/shuorenhua/vote/server
   python3 admin.py --db /var/lib/verna/verna.db import-items /path/to/items.jsonl
   ```

3. nginx：把 `nginx-api.conf.snippet` 里的 `location /api/ { ... }` 放进
   `verna.143143.eu.org` 的 443 `server {}` 块（本机 nginx 跑在名为 `nginx` 的容器里，
   配置是容器内的 `/etc/nginx/conf.d/verna.143143.eu.org.conf`）。
   `proxy_pass` 不带 URI，所以 `/api/...` 原样转发，`server.py` 自己去掉 `/api` 前缀。
   snippet 透传 `CF-Connecting-IP` 和 `X-Forwarded-For`，关掉 `proxy_buffering`。
   如果 nginx 在 http/server 层开了 `limit_req`，这个 location 用 `limit_req_dry_run on;`
   覆盖（只记录、不拒绝，且没有再写一条 `limit_req`）。本机 nginx 是 1.31.7，支持该指令；
   1.17.0 以前的版本要删掉这一行，改用 location 级 `limit_req` 放宽阈值。
   改完检查并重载：

   ```sh
   docker exec nginx nginx -t
   docker exec nginx nginx -s reload
   ```

4. 验证：

   ```sh
   curl -s http://127.0.0.1:8790/api/stats
   curl -s 'https://verna.143143.eu.org/api/stats'
   ```

5. 备份：SQLite 开了 WAL，直接 `cp` 可能拿到不一致的快照，用：

   ```sh
   sqlite3 /var/lib/verna/verna.db ".backup '/var/lib/verna/backup.db'"
   ```

   备份时别把 `ip_salt` 弄丢；换盐会让新旧 IP 哈希对不上（只影响同一人识别，不影响投票）。

6. 日志：`journalctl -u verna-vote -f`。请求日志默认关闭，排查时可在单元里给
   `ExecStart` 加 `--verbose` 后重启。

## 实现说明

- 每个请求一个 SQLite 连接，连接上设 `busy_timeout=10000`；库开 WAL。写操作用
  `BEGIN IMMEDIATE`，投稿的每日限额检查与插入在同一个写事务里。
- 时间统一存 UTC，ISO 格式 `YYYY-MM-DDTHH:MM:SSZ`。投稿限额按 UTC 日期统计。
- 投稿去重的 `text_hash` 会把文本里所有空白（空格、换行、制表符）去掉后再算，
  所以同样的文字换个换行不会重复入库。
- `items` 里的 `a_system`、`b_system`、`kind`、`probe_answer`、`meta` 只在库里，
  任何接口都不返回。
- 前端 `site/app.js` 默认请求同目录下的 `api/`；本机直接打开静态文件时接口不可用，
  会自动退回读 `items.json`，方便本地预览。
