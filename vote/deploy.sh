#!/bin/sh
# 把 vote/site/ 部署到 verna.143143.eu.org。部署副本里给 css/js 引用加内容哈希（Cloudflare 的浏览器缓存 TTL 是 4 小时），源文件不动。
set -e
SRC=/root/shuorenhua/vote/site
DST=/home/web/html/verna.143143.eu.org
cp "$SRC/style.css" "$SRC/app.js" "$SRC/items.json" "$DST/"
CV=$(md5sum "$SRC/style.css" | cut -c1-8); JV=$(md5sum "$SRC/app.js" | cut -c1-8)
sed -e "s#href=\"style\.css\"#href=\"style.css?v=$CV\"#" -e "s#src=\"app\.js\"#src=\"app.js?v=$JV\"#" "$SRC/index.html" > "$DST/index.html.tmp"
grep -q "style.css?v=$CV" "$DST/index.html.tmp" && grep -q "app.js?v=$JV" "$DST/index.html.tmp" || { echo "index.html 里没找到 style.css / app.js 引用，没部署"; rm -f "$DST/index.html.tmp"; exit 1; }
mv "$DST/index.html.tmp" "$DST/index.html"
docker exec nginx nginx -s reload  # nginx 开了 open_file_cache，换了文件要重载才会立刻生效
echo "已部署：style.css?v=$CV app.js?v=$JV"
