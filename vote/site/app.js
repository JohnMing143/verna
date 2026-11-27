(function () {
  "use strict";

  var $ = function (sel) {
    return document.querySelector(sel);
  };

  /* ------------------------------------------------------------------------
     Safe Storage (try/catch on all localStorage access)
     ------------------------------------------------------------------------ */
  var store = {
    get: function (key, fallback) {
      try {
        var raw = window.localStorage.getItem(key);
        return raw ? JSON.parse(raw) : fallback;
      } catch (e) {
        return fallback;
      }
    },
    set: function (key, val) {
      try {
        window.localStorage.setItem(key, JSON.stringify(val));
      } catch (e) {
        // Ignore storage quota or privacy-mode errors
      }
    }
  };

  var judge = store.get("verna.judge", null);
  if (!judge || !/^j[a-z0-9]{8,40}$/.test(String(judge))) {
    judge = "j" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
    store.set("verna.judge", judge);
  }
  var votes = store.get("verna.proto.votes", []);
  if (!Array.isArray(votes)) {
    votes = [];
  }
  var prefs = store.get("verna.prefs", { diff: false, quiet: false });
  if (!prefs || typeof prefs !== "object") {
    prefs = { diff: false, quiet: false };
  }

  var skips = store.get("verna.proto.skips", []);
  if (!Array.isArray(skips)) {
    skips = [];
  }
  var skippedMap = {};
  for (var skipIndex = 0; skipIndex < skips.length; skipIndex++) {
    if (typeof skips[skipIndex] === "string") {
      skippedMap[skips[skipIndex]] = true;
    }
  }

  var votedMap = {};
  function syncVotedMap() {
    votedMap = {};
    for (var vi = 0; vi < votes.length; vi++) {
      if (votes[vi] && votes[vi].item) {
        votedMap[votes[vi].item] = true;
      }
    }
  }
  syncVotedMap();

  /* ------------------------------------------------------------------------
     API Boundary — backend at /api/, items.json fallback for local preview
     ------------------------------------------------------------------------ */
  var API_BASE = "api/";
  var BATCH = 5;
  var fromParam = "";
  (function () {
    var match = /[?&]from=([^&#]*)/.exec(window.location.search || "");
    if (match) {
      try {
        fromParam = decodeURIComponent(match[1].replace(/\+/g, " "));
      } catch (e) {
        fromParam = "";
      }
    }
  })();
  var fromSent = false;
  var pending = store.get("verna.proto.pending", []);
  if (!Array.isArray(pending)) {
    pending = [];
  }
  var flushing = false;

  function savePending() {
    store.set("verna.proto.pending", pending);
  }

  function endpointFor(op) {
    if (op.kind === "vote") {
      return "vote";
    }
    if (op.kind === "unvote") {
      return "unvote";
    }
    if (op.kind === "skip") {
      return "skip";
    }
    return op.kind;
  }

  function apiJson(method, path, payload) {
    var opts = {
      method: method,
      headers: { Accept: "application/json" },
      cache: "no-store"
    };
    if (payload !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(payload);
    }
    return fetch(API_BASE + path, opts).then(function (res) {
      return res.text().then(function (text) {
        var json = null;
        try {
          json = text ? JSON.parse(text) : null;
        } catch (e) {
          json = null;
        }
        if (!res.ok || !json || json.ok !== true) {
          var err = new Error((json && json.error) || "HTTP " + res.status);
          err.status = res.status;
          err.code = json && json.error;
          throw err;
        }
        return json;
      });
    });
  }

  function fetchItems() {
    var url = "next?judge=" + encodeURIComponent(judge) + "&n=" + BATCH;
    if (fromParam && !fromSent) {
      url += "&from=" + encodeURIComponent(fromParam);
    }
    return apiJson("GET", url).then(function (data) {
      if (!Array.isArray(data.items)) {
        throw new Error("Invalid items format");
      }
      fromSent = true;
      flushPending();
      return data.items;
    });
  }

  function fetchLocalItems() {
    return fetch("items.json").then(function (res) {
      if (!res.ok) {
        throw new Error("HTTP " + res.status);
      }
      return res.json();
    });
  }

  function loadBatch() {
    return fetchItems().catch(function () {
      return fetchLocalItems();
    });
  }

  function enqueuePending(op) {
    if (!op || !op.kind) {
      return;
    }
    for (var i = pending.length - 1; i >= 0; i--) {
      var queued = pending[i];
      if (!queued) {
        continue;
      }
      if (queued.item !== op.item || queued.judge !== op.judge) {
        continue;
      }
      if (op.kind === "unvote") {
        if (queued.kind === "vote" || queued.kind === "unvote") {
          pending.splice(i, 1);
        }
      } else if (queued.kind === op.kind) {
        pending.splice(i, 1);
      }
    }
    pending.push(op);
    savePending();
  }

  function flushPending() {
    if (flushing || !pending.length) {
      return;
    }
    flushing = true;
    function step() {
      if (!pending.length) {
        flushing = false;
        return;
      }
      var op = pending[0];
      apiJson("POST", endpointFor(op), op).then(
        function () {
          var idx = pending.indexOf(op);
          if (idx >= 0) {
            pending.splice(idx, 1);
          }
          savePending();
          step();
        },
        function (err) {
          if (err && err.status && err.status < 500 && err.status !== 429) {
            var idx = pending.indexOf(op);
            if (idx >= 0) {
              pending.splice(idx, 1);
            }
            savePending();
            step();
            return;
          }
          flushing = false;
        }
      );
    }
    step();
  }

  function dropPendingFor(op) {
    if (!op || !op.item || !op.judge) {
      return;
    }
    for (var i = pending.length - 1; i >= 0; i--) {
      var queued = pending[i];
      if (!queued || queued.item !== op.item || queued.judge !== op.judge) {
        continue;
      }
      if (op.kind === "vote" || op.kind === "unvote") {
        if (queued.kind === "vote" || queued.kind === "unvote") {
          pending.splice(i, 1);
        }
      } else if (op.kind === "skip" && queued.kind === "skip") {
        pending.splice(i, 1);
      }
    }
    savePending();
  }

  function sendOrQueue(op) {
    apiJson("POST", endpointFor(op), op).then(
      function () {
        // The server now has the newest state; anything still queued for
        // this item would only overwrite it with stale data.
        dropPendingFor(op);
        flushPending();
      },
      function (err) {
        if (!err || !err.status || err.status >= 500 || err.status === 429) {
          enqueuePending(op);
        }
      }
    );
  }

  function submitVote(record) {
    var idx = -1;
    for (var i = votes.length - 1; i >= 0; i--) {
      if (votes[i] && votes[i].item === record.item && votes[i].at === record.at) {
        idx = i;
        break;
      }
    }
    if (idx >= 0) {
      votes[idx] = record;
    } else {
      votes.push(record);
    }
    store.set("verna.proto.votes", votes);
    syncVotedMap();

    sendOrQueue({
      kind: "vote",
      item: record.item,
      judge: record.judge,
      choice: record.choice,
      left: record.left,
      ms: record.ms,
      reasons: record.reasons || [],
      note: record.note || ""
    });
    return Promise.resolve(record);
  }

  function removeVoteByItem(itemId) {
    for (var i = votes.length - 1; i >= 0; i--) {
      if (votes[i] && votes[i].item === itemId) {
        votes.splice(i, 1);
        break;
      }
    }
    store.set("verna.proto.votes", votes);
    syncVotedMap();
  }

  function sendUnvote(itemId) {
    sendOrQueue({ kind: "unvote", item: itemId, judge: judge });
  }

  function sendSkip(itemId) {
    sendOrQueue({ kind: "skip", item: itemId, judge: judge });
  }

  /* ------------------------------------------------------------------------
     Constants & State
     ------------------------------------------------------------------------ */
  var GOOD = ["读着顺", "更简洁", "信息更完整", "保住了原来的语气"];
  var BAD = [
    "有空话、套话",
    "压得太狠，像拍电报",
    "丢了或改错了信息",
    "加了原文没有的内容",
    "语气或文体变了",
    "用词别扭"
  ];

  var queue = [];
  var itemsLoaded = false;
  var itemsLoading = false;
  var loadPromise = null;
  var exhausted = false;
  var cur = null;
  var shownAt = 0;
  var last = null;
  var activeView = "home";
  var toastTimer = null;
  var submitBusy = false;

  var reduceMotionQuery = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)")
    : null;

  function prefersReducedMotion() {
    return !!(reduceMotionQuery && reduceMotionQuery.matches);
  }

  function listenMedia(mq, fn) {
    if (!mq) {
      return;
    }
    if (mq.addEventListener) {
      mq.addEventListener("change", fn);
    } else if (mq.addListener) {
      mq.addListener(fn);
    }
  }

  /* ------------------------------------------------------------------------
     Text Utilities (preserving prototype behavior)
     ------------------------------------------------------------------------ */
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function plain(s) {
    return String(s).replace(/\*\*(.+?)\*\*/g, "$1");
  }

  function hanLen(s) {
    var m = plain(s).match(/[\u4e00-\u9fff]/g);
    return m ? m.length : 0;
  }

  function renderText(s, marks) {
    var out = [];
    var bold = false;
    var i = 0;
    var k = 0;
    var para = "";
    var t = String(s);

    function flush() {
      if (para.trim()) {
        out.push("<p>" + para + "</p>");
      }
      para = "";
    }

    while (i < t.length) {
      if (t[i] === "*" && t[i + 1] === "*") {
        para += bold ? "</strong>" : "<strong>";
        bold = !bold;
        i += 2;
        continue;
      }
      var ch = t[i];
      if (ch === "\n") {
        if (bold) {
          para += "</strong>";
        }
        flush();
        if (bold) {
          para += "<strong>";
        }
        i += 1;
        k += 1;
        continue;
      }
      var escaped = esc(ch);
      para += marks && marks[k] ? '<mark class="ins">' + escaped + "</mark>" : escaped;
      i += 1;
      k += 1;
    }
    if (bold) {
      para += "</strong>";
    }
    flush();
    return out.join("").replace(/<\/mark><mark class="ins">/g, "");
  }

  // Character-level LCS: highlight characters added in rewrite that are absent from source
  function newChars(src, cand) {
    var a = plain(src);
    var b = plain(cand);
    var n = a.length;
    var m = b.length;
    if (n * m > 2.5e6) {
      return null;
    }
    var dp = new Uint16Array((n + 1) * (m + 1));
    var W = m + 1;
    var i;
    var j;
    for (i = n - 1; i >= 0; i--) {
      for (j = m - 1; j >= 0; j--) {
        dp[i * W + j] =
          a[i] === b[j]
            ? dp[(i + 1) * W + j + 1] + 1
            : Math.max(dp[(i + 1) * W + j], dp[i * W + j + 1]);
      }
    }
    var mark = new Array(m);
    for (var q = 0; q < m; q++) {
      mark[q] = true;
    }
    var x = 0;
    var y = 0;
    while (x < n && y < m) {
      if (a[x] === b[y]) {
        mark[y] = false;
        x += 1;
        y += 1;
      } else if (dp[(x + 1) * W + y] >= dp[x * W + y + 1]) {
        x += 1;
      } else {
        y += 1;
      }
    }
    for (var r = 0; r < m; r++) {
      if (/\s/.test(b[r])) {
        mark[r] = false;
      }
    }
    return mark;
  }

  function shuffle(arr) {
    for (var i = arr.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  /* ------------------------------------------------------------------------
     Toast Helper
     ------------------------------------------------------------------------ */
  function showToast(msg) {
    var el = $("#toast");
    el.textContent = msg;
    el.classList.add("is-on");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      el.classList.remove("is-on");
    }, 1500);
  }

  /* ------------------------------------------------------------------------
     Review View Logic
     ------------------------------------------------------------------------ */
  function updateCount() {
    var el = $("#review-count");
    el.textContent = "已评 " + votes.length + " 题";
  }

  function setDockDone(isDone) {
    var dock = $("#vote-dock");
    var review = $("#view-review");
    if (isDone) {
      dock.classList.add("is-done");
      review.classList.add("has-dock-done");
    } else {
      dock.classList.remove("is-done");
      review.classList.remove("has-dock-done");
    }
    updateSkipVisibility();
  }

  function updateSkipVisibility() {
    var skipBtn = $("#btn-skip");
    if (!skipBtn) {
      return;
    }
    var done = $("#vote-dock").classList.contains("is-done");
    skipBtn.hidden = done || (queue.length <= 1 && exhausted);
  }

  function fillCandidate(sel, text) {
    var card = $(sel);
    card.querySelector(".char-count").textContent = hanLen(text) + " 字";
    var marks = prefs.diff ? newChars(cur.source, text) : null;
    card.querySelector(".cand-prose").innerHTML = renderText(text, marks);
  }

  /* Fetch the next batch and append anything this judge has not seen yet. */
  function refill() {
    if (loadPromise) {
      return loadPromise;
    }
    if (exhausted) {
      return Promise.resolve(false);
    }
    loadPromise = loadBatch().then(
      function (list) {
        loadPromise = null;
        var have = {};
        var i;
        for (i = 0; i < queue.length; i++) {
          if (queue[i] && queue[i].id) {
            have[queue[i].id] = true;
          }
        }
        for (i = 0; i < votes.length; i++) {
          if (votes[i] && votes[i].item) {
            have[votes[i].item] = true;
          }
        }
        for (i = 0; i < skips.length; i++) {
          if (typeof skips[i] === "string") {
            have[skips[i]] = true;
          }
        }

        var fresh = [];
        var rows = Array.isArray(list) ? list : [];
        for (i = 0; i < rows.length; i++) {
          var row = rows[i];
          if (!row || typeof row.id !== "string" || have[row.id]) {
            continue;
          }
          if (votedMap[row.id] || skippedMap[row.id]) {
            continue;
          }
          have[row.id] = true;
          fresh.push({
            id: row.id,
            genre: row.genre || "",
            source: row.source || "",
            a: row.a || "",
            b: row.b || ""
          });
        }
        if (!fresh.length) {
          exhausted = true;
        } else {
          queue = queue.concat(fresh);
        }
        return fresh.length > 0;
      },
      function (err) {
        loadPromise = null;
        throw err;
      }
    );
    return loadPromise;
  }

  function showLoadError() {
    cur = null;
    $("#review-item").hidden = true;
    $("#review-end").hidden = true;
    $("#vote-dock").hidden = true;
    $("#review-error").hidden = false;
  }

  function ensureMore() {
    if (loadPromise || exhausted) {
      return;
    }
    refill().then(
      function () {
        updateSkipVisibility();
        if (!cur) {
          showCurrentItem();
        }
      },
      function () {
        if (!cur) {
          showLoadError();
        }
      }
    );
  }

  function advance() {
    showCurrentItem();
    if (queue.length < 2 && !exhausted) {
      ensureMore();
    }
  }

  function showCurrentItem() {
    if (activeView !== "review") {
      return;
    }
    while (queue.length && (votedMap[queue[0].id] || skippedMap[queue[0].id])) {
      queue.shift();
    }

    $("#review-error").hidden = true;

    if (!queue.length) {
      cur = null;
      $("#review-item").hidden = true;
      $("#vote-dock").hidden = true;
      setDockDone(false);
      if (exhausted) {
        $("#review-end").hidden = false;
        $("#end-summary").textContent = "感谢你的评审。你一共评了 " + votes.length + " 题。";
      } else {
        $("#review-end").hidden = true;
        ensureMore();
      }
      return;
    }

    cur = queue[0];
    cur.flip = Math.random() < 0.5;
    var leftText = cur.flip ? cur.b : cur.a;
    var rightText = cur.flip ? cur.a : cur.b;

    $("#review-end").hidden = true;
    $("#review-item").hidden = false;
    $("#vote-dock").hidden = false;

    $("#item-genre").textContent = cur.genre || "";
    $("#src-len").textContent = hanLen(cur.source) + " 字";
    $("#src-text").innerHTML = renderText(cur.source, null);

    var srcCard = $("#src-card");
    srcCard.classList.remove("is-folded");
    $("#fold-label").textContent = "收起";
    $("#btn-fold").setAttribute("aria-expanded", "true");

    fillCandidate("#cand-a", leftText);
    fillCandidate("#cand-b", rightText);

    $("#cand-a").classList.remove("is-picked", "is-dim");
    $("#cand-b").classList.remove("is-picked", "is-dim");
    setDockDone(false);
    last = null;
    shownAt = Date.now();
    window.scrollTo(0, 0);
  }

  function ensureItemsLoaded() {
    if (itemsLoaded) {
      if (cur) {
        $("#review-item").hidden = false;
        $("#review-end").hidden = true;
        $("#review-error").hidden = true;
        $("#vote-dock").hidden = false;
        updateSkipVisibility();
      } else {
        showCurrentItem();
      }
      return;
    }
    if (itemsLoading) {
      return;
    }
    itemsLoading = true;
    refill().then(
      function () {
        itemsLoading = false;
        itemsLoaded = true;
        showCurrentItem();
      },
      function () {
        itemsLoading = false;
        showLoadError();
      }
    );
  }

  function getRecordedText(v) {
    if (v === "L") {
      return "已记录：你选了 A。";
    }
    if (v === "R") {
      return "已记录：你选了 B。";
    }
    return "已记录：原文更好。";
  }

  function handleVote(v) {
    if (!cur || $("#vote-dock").classList.contains("is-done")) {
      return;
    }
    var pick =
      v === "L"
        ? cur.flip
          ? "b"
          : "a"
        : v === "R"
          ? cur.flip
            ? "a"
            : "b"
          : v;

    last = {
      item: cur.id,
      judge: judge,
      choice: pick,
      left: cur.flip ? "b" : "a",
      ms: Math.min(Math.max(0, Date.now() - shownAt), 86400000),
      reasons: [],
      note: "",
      at: new Date().toISOString()
    };

    submitVote(last);
    updateCount();

    var recordedText = getRecordedText(v);

    if (prefs.quiet) {
      showToast(recordedText);
      queue.shift();
      last = null;
      advance();
      return;
    }

    var side = v === "L" ? "A" : v === "R" ? "B" : null;
    var other = side === "A" ? "B" : "A";

    if (side) {
      $(side === "A" ? "#cand-a" : "#cand-b").classList.add("is-picked");
      $(other === "A" ? "#cand-a" : "#cand-b").classList.add("is-dim");
    }

    $("#after-said").textContent = recordedText;

    var groups = side
      ? [
          [side + " 好在哪", GOOD, "good:"],
          [other + " 的问题", BAD, "bad:"]
        ]
      : [["两份改写的问题", BAD, "bad:"]];

    $("#reason-groups").innerHTML = groups
      .map(function (group) {
        var title = group[0];
        var list = group[1];
        var prefix = group[2];
        var chips = list
          .map(function (label) {
            return (
              '<button type="button" class="reason-chip" aria-pressed="false" data-r="' +
              esc(prefix + label) +
              '">' +
              esc(label) +
              "</button>"
            );
          })
          .join("");
        return (
          '<div class="reason-group"><h4>' +
          esc(title) +
          '</h4><div class="reason-chips">' +
          chips +
          "</div></div>"
        );
      })
      .join("");

    $("#note-input").value = "";
    setDockDone(true);
  }

  function handleNext() {
    if (!last || !$("#vote-dock").classList.contains("is-done")) {
      return;
    }
    var pressed = document.querySelectorAll('.reason-chip[aria-pressed="true"]');
    last.reasons = Array.prototype.map.call(pressed, function (btn) {
      return btn.getAttribute("data-r");
    });
    last.note = $("#note-input").value.trim();
    submitVote(last);
    if (queue.length) {
      queue.shift();
    }
    last = null;
    advance();
  }

  function handleUndo() {
    if (!last) {
      return;
    }
    var itemId = last.item;
    removeVoteByItem(itemId);
    sendUnvote(itemId);
    updateCount();
    last = null;
    setDockDone(false);
    $("#cand-a").classList.remove("is-picked", "is-dim");
    $("#cand-b").classList.remove("is-picked", "is-dim");
  }

  function handleSkip() {
    if (!cur) {
      return;
    }
    var itemId = cur.id;
    if (!skippedMap[itemId]) {
      skippedMap[itemId] = true;
      skips.push(itemId);
      store.set("verna.proto.skips", skips);
    }
    sendSkip(itemId);
    if (queue.length && queue[0].id === itemId) {
      queue.shift();
    }
    last = null;
    advance();
  }

  /* ------------------------------------------------------------------------
     Hash Routing (#/ and #/review)
     ------------------------------------------------------------------------ */
  function parseRoute() {
    var h = window.location.hash || "#/";
    if (h === "#/review" || h.indexOf("#/review/") === 0 || h.indexOf("#/review?") === 0) {
      return "review";
    }
    if (h === "#/submit" || h.indexOf("#/submit/") === 0 || h.indexOf("#/submit?") === 0) {
      return "submit";
    }
    return "home";
  }

  function applyRoute() {
    var nextView = parseRoute();
    activeView = nextView;

    var isReview = nextView === "review";
    var isSubmit = nextView === "submit";
    $("#view-home").hidden = nextView !== "home";
    $("#view-review").hidden = !isReview;
    $("#view-submit").hidden = !isSubmit;
    $("#header-cta").hidden = isReview;
    $("#header-nav-submit").hidden = isSubmit;
    $("#review-count").hidden = !isReview;

    if (isReview) {
      updateCount();
      heroCanvas.stop();
      specimens.pause();
      ensureItemsLoaded();
    } else if (isSubmit) {
      $("#vote-dock").hidden = true;
      heroCanvas.stop();
      specimens.pause();
      if (!$("#submit-done").hidden) {
        resetSubmitForm();
      } else {
        updateSubmitCount();
      }
      window.scrollTo(0, 0);
    } else {
      $("#vote-dock").hidden = true;
      heroCanvas.sync();
      specimens.sync();
      window.scrollTo(0, 0);
    }
  }

  /* ------------------------------------------------------------------------
     Landing Hero Canvas — Subtle Architectural Proof-Grid & Light Drift
     Pauses automatically off-viewport, on tab hidden, in review, or reduced motion.
     ------------------------------------------------------------------------ */
  var heroCanvas = (function () {
    var canvas = $("#hero-canvas");
    var heroEl = $("#hero");
    if (!canvas || !heroEl || !canvas.getContext) {
      return {
        sync: function () {},
        stop: function () {}
      };
    }

    var ctx = canvas.getContext("2d");
    var rafId = null;
    var inViewport = true;
    var width = 0;
    var height = 0;
    var dpr = 1;
    var pointerX = 0.5;
    var pointerY = 0.42;
    var targetX = 0.5;
    var targetY = 0.42;
    var darkQuery = window.matchMedia
      ? window.matchMedia("(prefers-color-scheme: dark)")
      : null;

    var marks = [];
    for (var i = 0; i < 18; i++) {
      marks.push({
        x: ((i * 37) % 100) / 100,
        y: ((i * 53 + 11) % 100) / 100,
        size: 4 + (i % 3) * 2.5,
        phase: i * 0.7,
        speed: 0.25 + (i % 4) * 0.1,
        accent: i % 5 === 0
      });
    }

    function isDark() {
      return !!(darkQuery && darkQuery.matches);
    }

    function resize() {
      var rect = heroEl.getBoundingClientRect();
      width = Math.max(1, Math.floor(rect.width));
      height = Math.max(1, Math.floor(rect.height));
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawFrame(performance.now ? performance.now() : 0);
    }

    function drawFrame(now) {
      var t = now * 0.001;
      var dark = isDark();

      pointerX += (targetX - pointerX) * 0.05;
      pointerY += (targetY - pointerY) * 0.05;

      ctx.clearRect(0, 0, width, height);

      // Subtle architectural grid
      var step = width < 640 ? 44 : 64;
      var gridColor = dark ? "rgba(255,255,255,0.036)" : "rgba(20,19,17,0.042)";
      var offsetX = (pointerX - 0.5) * -10;
      var offsetY = (pointerY - 0.5) * -6;

      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      var x;
      for (x = (offsetX % step) - step; x < width + step; x += step) {
        var gx = Math.round(x) + 0.5;
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, height);
      }
      var y;
      for (y = (offsetY % step) - step; y < height + step; y += step) {
        var gy = Math.round(y) + 0.5;
        ctx.moveTo(0, gy);
        ctx.lineTo(width, gy);
      }
      ctx.stroke();

      // Delicate editorial registration crosshairs drifting quietly
      for (var i = 0; i < marks.length; i++) {
        var m = marks[i];
        var mx = m.x * width + Math.sin(t * m.speed + m.phase) * 14 + offsetX * 1.4;
        var my = m.y * height + Math.cos(t * m.speed * 0.8 + m.phase) * 10 + offsetY * 1.4;
        var pulse = 0.45 + 0.55 * Math.sin(t * 0.7 + m.phase);
        var alpha = m.accent ? (dark ? 0.34 : 0.28) * pulse : (dark ? 0.14 : 0.13) * pulse;

        ctx.strokeStyle = m.accent
          ? (dark ? "rgba(222,111,80," : "rgba(184,66,37,") + alpha.toFixed(3) + ")"
          : (dark ? "rgba(237,236,232," : "rgba(20,19,17,") + alpha.toFixed(3) + ")";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mx - m.size, my);
        ctx.lineTo(mx + m.size, my);
        ctx.moveTo(mx, my - m.size);
        ctx.lineTo(mx, my + m.size);
        ctx.stroke();
      }
    }

    function loop(now) {
      drawFrame(now);
      rafId = window.requestAnimationFrame(loop);
    }

    function shouldRun() {
      return (
        activeView === "home" &&
        inViewport &&
        !document.hidden &&
        !prefersReducedMotion()
      );
    }

    function start() {
      if (rafId !== null) {
        return;
      }
      resize();
      if (shouldRun()) {
        rafId = window.requestAnimationFrame(loop);
      }
    }

    function stop() {
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId);
        rafId = null;
      }
    }

    function sync() {
      if (shouldRun()) {
        start();
      } else {
        stop();
        if (activeView === "home") {
          resize();
        }
      }
    }

    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(
        function (entries) {
          if (entries && entries[0]) {
            inViewport = entries[0].isIntersecting;
            sync();
          }
        },
        { threshold: 0.02 }
      );
      io.observe(heroEl);
    }

    heroEl.addEventListener("mousemove", function (e) {
      if (prefersReducedMotion()) {
        return;
      }
      var rect = heroEl.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        targetX = (e.clientX - rect.left) / rect.width;
        targetY = (e.clientY - rect.top) / rect.height;
      }
    });

    window.addEventListener("resize", function () {
      if (activeView === "home") {
        resize();
      }
    });

    listenMedia(darkQuery, function () {
      if (activeView === "home") {
        drawFrame(performance.now ? performance.now() : 0);
      }
    });

    resize();

    return {
      sync: sync,
      stop: stop
    };
  })();

  /* ------------------------------------------------------------------------
     Landing Specimens (Problem 01 Redaction & Problem 02 Expansion)
     ------------------------------------------------------------------------ */
  var specimens = (function () {
    var redactEl = $("#specimen-redact");
    var expandEl = $("#specimen-expand");
    var timers = { redact: null, expand: null };
    var visible = { redact: false, expand: false };
    var manualHold = { redact: false, expand: false };

    function setFinalStatic() {
      if (redactEl) {
        redactEl.classList.add("is-edited");
      }
      if (expandEl) {
        expandEl.classList.add("is-expanded");
      }
    }

    function clearTimers() {
      if (timers.redact) {
        clearInterval(timers.redact);
        timers.redact = null;
      }
      if (timers.expand) {
        clearInterval(timers.expand);
        timers.expand = null;
      }
    }

    function startRedactCycle() {
      if (!redactEl || timers.redact || manualHold.redact) {
        return;
      }
      setTimeout(function () {
        if (visible.redact && !manualHold.redact && !prefersReducedMotion()) {
          redactEl.classList.add("is-edited");
        }
      }, 650);
      timers.redact = setInterval(function () {
        if (visible.redact && !document.hidden && activeView === "home") {
          redactEl.classList.toggle("is-edited");
        }
      }, 3600);
    }

    function startExpandCycle() {
      if (!expandEl || timers.expand || manualHold.expand) {
        return;
      }
      setTimeout(function () {
        if (visible.expand && !manualHold.expand && !prefersReducedMotion()) {
          expandEl.classList.add("is-expanded");
        }
      }, 850);
      timers.expand = setInterval(function () {
        if (visible.expand && !document.hidden && activeView === "home") {
          expandEl.classList.toggle("is-expanded");
        }
      }, 3800);
    }

    function sync() {
      clearTimers();
      if (prefersReducedMotion()) {
        setFinalStatic();
        return;
      }
      if (activeView !== "home" || document.hidden) {
        return;
      }
      if (visible.redact) {
        startRedactCycle();
      }
      if (visible.expand) {
        startExpandCycle();
      }
    }

    function bindToggle(el, key, cls) {
      if (!el) {
        return;
      }
      function toggle() {
        manualHold[key] = true;
        if (timers[key]) {
          clearInterval(timers[key]);
          timers[key] = null;
        }
        el.classList.toggle(cls);
      }
      el.addEventListener("click", toggle);
      el.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggle();
        }
      });
    }

    bindToggle(redactEl, "redact", "is-edited");
    bindToggle(expandEl, "expand", "is-expanded");

    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(
        function (entries) {
          for (var i = 0; i < entries.length; i++) {
            var entry = entries[i];
            if (entry.target === redactEl) {
              visible.redact = entry.isIntersecting;
            } else if (entry.target === expandEl) {
              visible.expand = entry.isIntersecting;
            }
          }
          sync();
        },
        { threshold: 0.3 }
      );
      if (redactEl) {
        io.observe(redactEl);
      }
      if (expandEl) {
        io.observe(expandEl);
      }
    } else {
      visible.redact = true;
      visible.expand = true;
      setFinalStatic();
    }

    return {
      sync: sync,
      pause: clearTimers
    };
  })();

  /* ------------------------------------------------------------------------
     Landing Scroll Reveal
     ------------------------------------------------------------------------ */
  function initReveals() {
    var nodes = document.querySelectorAll(".reveal");
    if (!nodes.length) {
      return;
    }
    if (prefersReducedMotion() || !("IntersectionObserver" in window)) {
      for (var i = 0; i < nodes.length; i++) {
        nodes[i].classList.add("is-visible");
      }
      return;
    }
    var io = new IntersectionObserver(
      function (entries) {
        for (var j = 0; j < entries.length; j++) {
          if (entries[j].isIntersecting) {
            entries[j].target.classList.add("is-visible");
            io.unobserve(entries[j].target);
          }
        }
      },
      { threshold: 0.12 }
    );
    for (var k = 0; k < nodes.length; k++) {
      io.observe(nodes[k]);
    }
  }

  /* ------------------------------------------------------------------------
     Copy Link (Clipboard API + WeChat WebView fallback)
     ------------------------------------------------------------------------ */
  function copyShareLink() {
    var url = window.location.href.split("#")[0];
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(
        function () {
          showToast("链接已复制");
        },
        function () {
          fallbackCopy(url);
        }
      );
    } else {
      fallbackCopy(url);
    }
  }

  function fallbackCopy(text) {
    try {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      ta.setSelectionRange(0, text.length);
      var ok = document.execCommand("copy");
      document.body.removeChild(ta);
      if (ok) {
        showToast("链接已复制");
        return;
      }
    } catch (e) {
      // Fallback below
    }
    showToast(text);
  }

  /* ------------------------------------------------------------------------
     Submission Page
     ------------------------------------------------------------------------ */
  function updateSubmitCount() {
    var area = $("#sub-text");
    if (!area) {
      return;
    }
    var value = area.value || "";
    $("#sub-count").textContent = "已输入 " + hanLen(value) + " 字";
    $("#sub-limit-hint").hidden = value.length <= 3000;
  }

  function showSubmitError(message) {
    $("#sub-error").textContent = message || "";
  }

  function resetSubmitForm() {
    var form = $("#submit-form");
    if (!form) {
      return;
    }
    form.reset();
    showSubmitError("");
    $("#sub-go").disabled = true;
    $("#submit-done").hidden = true;
    $("#submit-form").hidden = false;
    updateSubmitCount();
  }

  function handleSubmitForm(event) {
    event.preventDefault();
    if (submitBusy) {
      return;
    }
    var text = $("#sub-text").value.trim();
    if (text.length > 3000) {
      showSubmitError("原文超过 3000 字");
      return;
    }
    if (hanLen(text) < 50) {
      showSubmitError("原文不足 50 字");
      return;
    }
    if (!$("#sub-consent").checked) {
      return;
    }
    submitBusy = true;
    var button = $("#sub-go");
    button.disabled = true;
    showSubmitError("");
    apiJson("POST", "submit", {
      judge: judge,
      text: text,
      model: $("#sub-model").value,
      scene: $("#sub-scene").value.trim(),
      nickname: $("#sub-nickname").value.trim(),
      consent: true
    }).then(
      function () {
        submitBusy = false;
        button.disabled = !$("#sub-consent").checked;
        $("#submit-form").hidden = true;
        $("#submit-done").hidden = false;
        window.scrollTo(0, 0);
      },
      function (err) {
        submitBusy = false;
        button.disabled = !$("#sub-consent").checked;
        var code = err && err.code;
        var message = "提交失败，请稍后重试";
        if (code === "too_short") {
          message = "原文不足 50 字";
        } else if (code === "too_long") {
          message = "原文超过 3000 字";
        } else if (code === "duplicate") {
          message = "这段文字已经有人投过了";
        } else if (code === "limit") {
          message = "今天的投稿次数已达上限";
        }
        showSubmitError(message);
      }
    );
  }

  /* ------------------------------------------------------------------------
     Event Bindings
     ------------------------------------------------------------------------ */
  $("#btn-learn").addEventListener("click", function () {
    var target = $("#problem");
    if (target && target.scrollIntoView) {
      target.scrollIntoView({
        behavior: prefersReducedMotion() ? "auto" : "smooth",
        block: "start"
      });
    }
  });

  $("#btn-skip").addEventListener("click", handleSkip);

  $("#btn-fold").addEventListener("click", function () {
    var card = $("#src-card");
    var folded = card.classList.toggle("is-folded");
    $("#fold-label").textContent = folded ? "展开" : "收起";
    this.setAttribute("aria-expanded", folded ? "false" : "true");
  });

  $("#opt-diff").checked = !!prefs.diff;
  $("#opt-diff").addEventListener("change", function () {
    prefs.diff = this.checked;
    store.set("verna.prefs", prefs);
    if (cur) {
      fillCandidate("#cand-a", cur.flip ? cur.b : cur.a);
      fillCandidate("#cand-b", cur.flip ? cur.a : cur.b);
    }
  });

  $("#opt-quiet").checked = !!prefs.quiet;
  $("#opt-quiet").addEventListener("change", function () {
    prefs.quiet = this.checked;
    store.set("verna.prefs", prefs);
  });

  $("#btn-next").addEventListener("click", handleNext);
  $("#btn-undo").addEventListener("click", handleUndo);
  $("#btn-copy").addEventListener("click", copyShareLink);

  $("#submit-form").addEventListener("submit", handleSubmitForm);
  $("#sub-text").addEventListener("input", function () {
    updateSubmitCount();
    if ($("#sub-error").textContent) {
      showSubmitError("");
    }
  });
  $("#sub-consent").addEventListener("change", function () {
    $("#sub-go").disabled = submitBusy || !this.checked;
  });
  $("#sub-again").addEventListener("click", resetSubmitForm);

  document.addEventListener("click", function (e) {
    var btn = e.target && e.target.closest ? e.target.closest("[data-v],.reason-chip") : null;
    if (!btn) {
      return;
    }
    if (btn.hasAttribute("data-v")) {
      handleVote(btn.getAttribute("data-v"));
    } else if (btn.classList.contains("reason-chip")) {
      var pressed = btn.getAttribute("aria-pressed") === "true";
      btn.setAttribute("aria-pressed", pressed ? "false" : "true");
    }
  });

  document.addEventListener("keydown", function (e) {
    if (activeView !== "review" || !cur) {
      return;
    }
    if (e.isComposing || e.keyCode === 229) {
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) {
      return;
    }
    var tag = e.target && e.target.tagName ? e.target.tagName : "";
    if (/INPUT|TEXTAREA/.test(tag) && e.key !== "Enter") {
      return;
    }
    var done = $("#vote-dock").classList.contains("is-done");
    if (!done && e.key === "ArrowLeft") {
      e.preventDefault();
      handleVote("L");
    } else if (!done && e.key === "ArrowRight") {
      e.preventDefault();
      handleVote("R");
    } else if (!done && e.key === "ArrowDown") {
      e.preventDefault();
      handleVote("src");
    } else if (done && e.key === "Enter") {
      e.preventDefault();
      handleNext();
    }
  });

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      heroCanvas.stop();
      specimens.pause();
    } else if (activeView === "home") {
      heroCanvas.sync();
      specimens.sync();
    }
  });

  listenMedia(reduceMotionQuery, function () {
    heroCanvas.sync();
    specimens.sync();
  });

  window.addEventListener("hashchange", applyRoute);

  /* ------------------------------------------------------------------------
     Init
     ------------------------------------------------------------------------ */
  initReveals();
  updateCount();
  updateSubmitCount();
  applyRoute();
})();
