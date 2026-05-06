/*
 * LectureMind Copilot — client-side panel.
 *
 * Vanilla JS, no build step.  Loaded on every lecture report page via
 * ``<script defer src="/static/copilot.js"></script>``.  The rendered
 * template places a ``<aside id="cp-panel" data-bv="..." data-title="...">``
 * stub at the end of <body>; that's the single anchor this file
 * discovers on DOMContentLoaded.
 *
 * Architecture:
 * 1. CopilotPanel owns DOM, chips, and message history (capped at 10).
 * 2. streamAsk() does POST + ReadableStream SSE parsing (EventSource
 *    cannot POST; see D44).
 * 3. Anchor rendering = regex post-process on the accumulated answer
 *    text; invalid anchors wrapped by the server are rendered as
 *    .cp-anchor-invalid (no click handler).
 * 4. Pin buttons on chapter/frame/point cards add references; any
 *    mouse-up selection inside <main> surfaces a floating pin.
 *
 * Optional dep: ``window.marked`` (loaded via CDN in the template).
 * When absent we fall back to plaintext rendering so the page still
 * works if the CDN is blocked.
 */

(function () {
  "use strict";

  const MAX_HISTORY_TURNS = 10;
  const ASK_ENDPOINT = "/api/copilot/ask";
  const WEB_TOGGLE_KEY = "lecturemind_copilot_use_web";
  const STATE_KEY_PREFIX = "lecturemind_copilot_state_";
  const STATE_VERSION = 1;

  // ------------------------------------------------------------------
  // Utility: SSE parsing over fetch()
  // ------------------------------------------------------------------

  async function* streamAsk(body, signal) {
    const resp = await fetch(ASK_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      credentials: "same-origin",
      signal,
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`HTTP ${resp.status}: ${text.slice(0, 200) || resp.statusText}`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const parsed = parseSseChunk(chunk);
        if (parsed) yield parsed;
      }
    }
    if (buffer.trim()) {
      const parsed = parseSseChunk(buffer);
      if (parsed) yield parsed;
    }
  }

  function parseSseChunk(chunk) {
    let event = "message";
    const dataLines = [];
    for (const raw of chunk.split(/\r?\n/)) {
      if (!raw) continue;
      if (raw.startsWith(":")) continue; // comment
      const sep = raw.indexOf(":");
      const field = sep === -1 ? raw : raw.slice(0, sep);
      const val = sep === -1 ? "" : raw.slice(sep + 1).replace(/^ /, "");
      if (field === "event") event = val;
      else if (field === "data") dataLines.push(val);
    }
    if (!dataLines.length) return null;
    const rawData = dataLines.join("\n");
    let data;
    try { data = JSON.parse(rawData); } catch { data = { raw: rawData }; }
    return { event, data };
  }

  function friendlyErrorMessage(msg) {
    const text = String(msg || "");
    if (
      text.includes("AllocationQuota.FreeTierOnly") ||
      text.toLowerCase().includes("free tier") ||
      text.includes("额度")
    ) {
      return "当前 Qwen/DashScope 模型免费额度已用尽，所以这次真实问答没有执行。请在 DashScope 控制台关闭 use free tier only / 切换到付费额度，或临时换一个还有额度的模型后再试。";
    }
    if (text.includes("HTTP 403") || text.includes("Error code: 403")) {
      return "模型服务返回 403，通常是 API Key 权限、额度或控制台计费模式问题。请检查 .env 中的 DashScope 配置和控制台额度。";
    }
    return text || "unknown";
  }

  // ------------------------------------------------------------------
  // Anchor rendering (post-process the accumulated answer string)
  // ------------------------------------------------------------------

  // Matches [t=12:34], [F12], [Ch3] plus the [⚠ t=12:34] invalid form.
  const ANCHOR_REGEX = /\[(⚠\s)?(t=\d{1,2}:\d{2}|F\d+|Ch\d+)\]/g;

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function refKindLabel(kind) {
    if (kind === "chapter") return "章节";
    if (kind === "frame") return "图片";
    if (kind === "selection") return "文本";
    return "引用";
  }

  function refSummary(ref) {
    if (!ref) return "";
    if (ref.text) return ref.text;
    if (ref.kind === "chapter") return `章节 ${ref.id}`;
    if (ref.kind === "frame") return `关键帧 F${ref.id}`;
    return ref.label || ref.kind || "";
  }

  function hostPreviewText(host) {
    const clone = host.cloneNode(true);
    clone.querySelectorAll(".cp-pin").forEach((el) => el.remove());
    return clone.textContent.replace(/\s+/g, " ").trim().slice(0, 180);
  }

  function ensureLatestCopilotCss() {
    if (document.getElementById("cp-live-css")) return;
    const link = document.createElement("link");
    link.id = "cp-live-css";
    link.rel = "stylesheet";
    link.href = "/static/copilot.css";
    document.head.appendChild(link);
  }

  function safeStorageGet(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  }

  function safeStorageSet(key, value) {
    try { localStorage.setItem(key, value); } catch {}
  }

  function safeStorageRemove(key) {
    try { localStorage.removeItem(key); } catch {}
  }

  function storedRef(ref) {
    if (!ref || typeof ref !== "object") return null;
    const kind = String(ref.kind || "").trim();
    if (!kind) return null;
    return {
      kind,
      id: ref.id != null ? ref.id : null,
      text: ref.text || null,
      label: ref.label || null,
    };
  }

  function storedHistoryTurn(turn) {
    if (!turn || typeof turn !== "object") return null;
    const role = String(turn.role || "").trim().toLowerCase();
    const content = String(turn.content || "").trim();
    if (!content || !["user", "assistant"].includes(role)) return null;
    const out = { role, content };
    if (role === "user" && Array.isArray(turn.refs)) {
      const refs = turn.refs.map(storedRef).filter(Boolean);
      if (refs.length) out.refs = refs;
    }
    return out;
  }

  function replaceAnchors(html) {
    return html.replace(ANCHOR_REGEX, (m, warn, token) => {
      if (warn) {
        return `<span class="cp-anchor-invalid" title="锚点越界">[⚠ ${escapeHtml(token)}]</span>`;
      }
      let kind, id;
      if (token.startsWith("t=")) { kind = "t";  id = token.slice(2); }
      else if (token.startsWith("F"))  { kind = "F";  id = token.slice(1); }
      else if (token.startsWith("Ch")) { kind = "Ch"; id = token.slice(2); }
      else { return escapeHtml(m); }
      return `<a href="#" class="cp-anchor" data-anchor-kind="${kind}" ` +
             `data-anchor-id="${escapeHtml(id)}">[${escapeHtml(token)}]</a>`;
    });
  }

  function renderMarkdown(text) {
    text = transformSections(text);
    if (window.marked && typeof window.marked.parse === "function") {
      try {
        const html = window.marked.parse(text, { breaks: true, gfm: true });
        return replaceAnchors(html);
      } catch (err) {
        // fall through to plaintext
      }
    }
    // Very small plaintext fallback: newlines → <br>.
    return replaceAnchors(escapeHtml(text).replace(/\n/g, "<br>"));
  }

  const SECTION_LABELS = {
    evidence: "讲义证据",
    background: "背景补全",
    extension: "延伸理解",
    deep_dive: "原理深挖",
    application: "应用举例",
    boundary: "边界说明",
    offtopic: "问题引导",
  };
  const SECTION_RE = /^\s*\[\[\s*([a-zA-Z_]+)\s*\]\]\s*$/gm;

  function transformSections(text) {
    const matches = [...text.matchAll(SECTION_RE)];
    if (!matches.length) return text;
    let out = "";
    for (let i = 0; i < matches.length; i += 1) {
      const m = matches[i];
      const type = String(m[1] || "").toLowerCase();
      const label = SECTION_LABELS[type] || type;
      const start = m.index + m[0].length;
      const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
      const body = text.slice(start, end).trim();
      out += `<section class="cp-section cp-section-${escapeHtml(type)}"><h4>${escapeHtml(label)}</h4>\n\n${body}</section>\n\n`;
    }
    return out.trim();
  }

  // ------------------------------------------------------------------
  // Anchor click handler (flashes target, opens Bilibili on Shift+t)
  // ------------------------------------------------------------------

  function timestampToSeconds(value) {
    const m = /^(\d{1,2}):(\d{2})$/.exec(value);
    if (!m) return null;
    return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
  }

  function flashElement(el) {
    if (!el) return;
    el.classList.add("cp-anchor-flash");
    setTimeout(() => el.classList.remove("cp-anchor-flash"), 1600);
  }

  function findAnchorTarget(kind, id, bv) {
    if (kind === "Ch") {
      return document.getElementById("chapter-" + id);
    }
    if (kind === "F") {
      return document.querySelector(
        `[data-ref-kind="frame"][data-ref-id="${id}"]`
      );
    }
    if (kind === "t") {
      const secs = timestampToSeconds(id);
      if (secs == null) return null;
      // Find the chapter whose [start, end] contains this ts.
      const chapters = document.querySelectorAll("[data-ref-kind='chapter']");
      let best = null;
      for (const el of chapters) {
        const start = parseInt(el.dataset.chStart || "0", 10);
        const end   = parseInt(el.dataset.chEnd   || "0", 10);
        if (secs >= start && secs <= end) { best = el; break; }
      }
      return best;
    }
    return null;
  }

  function onAnchorClick(evt) {
    const a = evt.target.closest && evt.target.closest(".cp-anchor");
    if (!a) return;
    evt.preventDefault();
    const kind = a.dataset.anchorKind;
    const id = a.dataset.anchorId;
    const bv = document.body.dataset.bv ||
               (document.getElementById("cp-panel") &&
                document.getElementById("cp-panel").dataset.bv) || "";
    if (kind === "t" && evt.shiftKey) {
      const secs = timestampToSeconds(id);
      const video = document.querySelector("video");
      if (secs != null && video) {
        video.currentTime = secs;
        video.play().catch(() => {});
        flashElement(video);
        return;
      }
    }
    const target = findAnchorTarget(kind, id, bv);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      flashElement(target);
    }
  }

  function initImageLightbox() {
    if (document.querySelector(".cp-lightbox")) return;
    const box = document.createElement("div");
    box.className = "cp-lightbox";
    box.innerHTML = `
      <button type="button" class="cp-lightbox-close" aria-label="关闭">×</button>
      <img class="cp-lightbox-img" alt="" />
      <div class="cp-lightbox-caption"></div>
    `;
    document.body.appendChild(box);
    const img = box.querySelector(".cp-lightbox-img");
    const caption = box.querySelector(".cp-lightbox-caption");
    const open = (src, alt, text) => {
      if (!src) return;
      img.src = src;
      img.alt = alt || "";
      caption.textContent = text || alt || "";
      box.classList.add("cp-lightbox-open");
    };
    const close = () => {
      box.classList.remove("cp-lightbox-open");
      img.removeAttribute("src");
      caption.textContent = "";
    };
    box.addEventListener("click", (evt) => {
      if (evt.target === box || evt.target.closest(".cp-lightbox-close")) close();
    });
    document.addEventListener("keydown", (evt) => {
      if (evt.key === "Escape" && box.classList.contains("cp-lightbox-open")) close();
    });
    document.addEventListener("click", (evt) => {
      const target = evt.target;
      if (!target || !target.closest) return;
      if (target.closest(".cp-aside") || target.closest(".cp-lightbox")) return;
      const cover = target.closest(".cover-img");
      if (cover) {
        evt.preventDefault();
        const bg = getComputedStyle(cover).backgroundImage || "";
        const match = /^url\(["']?(.*?)["']?\)$/.exec(bg);
        if (!match) return;
        open(match[1], "cover", "封面");
        return;
      }
      const clickedImg = target.closest("img");
      if (!clickedImg) return;
      if (!clickedImg.closest("main") && !clickedImg.closest(".cover")) return;
      evt.preventDefault();
      const figure = clickedImg.closest("figure");
      const text = figure ? figure.textContent.replace(/\s+/g, " ").trim() : (clickedImg.alt || "");
      open(clickedImg.currentSrc || clickedImg.src, clickedImg.alt || "", text);
    }, true);
  }

  // ------------------------------------------------------------------
  // CopilotPanel
  // ------------------------------------------------------------------

  class CopilotPanel {
    constructor(stub) {
      this.bv = stub.dataset.bv || "";
      this.title = stub.dataset.title || "";
      this.stub = stub;
      this.chips = []; // [{id, kind, id?, text?, label}]
      this.history = []; // [{role, content}]
      this.storageKey = STATE_KEY_PREFIX + (this.bv || "unknown");
      this.abortController = null;
      this.currentAssistantEl = null;
      this.currentAssistantRaw = "";
      this.currentTraceEl = null;
      this._build();
    }

    _build() {
      this.fab = document.createElement("button");
      this.fab.className = "cp-fab";
      this.fab.setAttribute("aria-label", "打开讲义助手");
      this.fab.setAttribute("aria-expanded", "false");
      this.fab.innerHTML = "＋";
      document.body.appendChild(this.fab);

      this.panel = document.createElement("aside");
      this.panel.className = "cp-aside";
      this.panel.innerHTML = `
        <div class="cp-header">
          <div>
            <div class="cp-header-title">讲义助手</div>
            <div class="cp-header-sub" data-role="sub"></div>
          </div>
          <div class="cp-header-actions">
            <button class="cp-clear" type="button" aria-label="清空对话">清空</button>
            <button class="cp-close" aria-label="关闭">✕</button>
          </div>
        </div>
        <div class="cp-status" data-role="status"></div>
        <div class="cp-messages" data-role="messages"></div>
        <div class="cp-composer">
          <div class="cp-chips" data-role="chips"></div>
          <div class="cp-composer-row">
            <textarea class="cp-textarea" rows="2"
              placeholder="问这段讲义的任何问题… 引用会显示在上方"></textarea>
            <button class="cp-web-toggle" type="button" aria-pressed="false" title="开启联网补充">🌐</button>
            <button class="cp-send">发送</button>
          </div>
        </div>
      `;
      document.body.appendChild(this.panel);

      this.elStatus   = this.panel.querySelector("[data-role=status]");
      this.elChips    = this.panel.querySelector("[data-role=chips]");
      this.elMessages = this.panel.querySelector("[data-role=messages]");
      this.elTextarea = this.panel.querySelector(".cp-textarea");
      this.elWebToggle = this.panel.querySelector(".cp-web-toggle");
      this.elSend     = this.panel.querySelector(".cp-send");
      this.elClear    = this.panel.querySelector(".cp-clear");
      this.elSub      = this.panel.querySelector("[data-role=sub]");
      this.elSub.textContent = this.title ? `${this.title} · ${this.bv}` : this.bv;
      this.useWeb = localStorage.getItem(WEB_TOGGLE_KEY) === "1";
      this._renderWebToggle();
      this._renderChips();
      this._restoreState();

      // Floating selection pin.
      this.selPin = document.createElement("button");
      this.selPin.className = "cp-selection-pin";
      this.selPin.type = "button";
      this.selPin.textContent = "📌 发给助手";
      document.body.appendChild(this.selPin);

      this._bindEvents();
    }

    _bindEvents() {
      this.fab.addEventListener("click", () => this.toggle());
      this.panel.querySelector(".cp-close").addEventListener("click", () => this.close());
      this.elClear.addEventListener("click", () => this.clearConversation());
      this.elSend.addEventListener("click", () => this.send());
      this.elWebToggle.addEventListener("click", () => {
        this.useWeb = !this.useWeb;
        localStorage.setItem(WEB_TOGGLE_KEY, this.useWeb ? "1" : "0");
        this._renderWebToggle();
      });
      this.elTextarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.send();
        }
      });
      this.elTextarea.addEventListener("input", () => this._persistState());

      // Pin button clicks (delegated).
      document.addEventListener("click", (e) => {
        const btn = e.target.closest && e.target.closest(".cp-pin");
        if (btn) {
          e.preventDefault();
          const host = btn.closest("[data-ref-kind]");
          if (host) this.addChipFromHost(host);
          return;
        }
        const frameId = e.target.closest && e.target.closest(".frame-id");
        if (frameId) {
          const host = frameId.closest('[data-ref-kind="frame"]');
          if (host) {
            e.preventDefault();
            this.addChipFromHost(host);
            return;
          }
        }
        // Selection pin click.
        if (e.target === this.selPin) {
          const text = window.getSelection && window.getSelection().toString().trim();
          if (text) {
            this.addChip({ kind: "selection", text, label: "「" + text.slice(0, 24) + "」" });
          }
          this.hideSelectionPin();
          return;
        }
      });
      document.addEventListener("dblclick", (e) => {
        const host = e.target.closest && e.target.closest('[data-ref-kind="frame"]');
        if (!host || this.panel.contains(host)) return;
        if (e.target.closest("a, button, summary, details")) return;
        e.preventDefault();
        this.addChipFromHost(host);
      });

      // Anchor click delegation (scoped to assistant messages only).
      this.elMessages.addEventListener("click", onAnchorClick);

      // Selection change: show floating pin near the end of the selection.
      document.addEventListener("selectionchange", () => {
        const sel = window.getSelection && window.getSelection();
        if (!sel || sel.isCollapsed || !sel.toString().trim()) {
          this.hideSelectionPin();
          return;
        }
        // Only honour selections inside <main>, not inside the panel.
        const anchorNode = sel.anchorNode;
        if (!anchorNode || this.panel.contains(anchorNode)) {
          this.hideSelectionPin();
          return;
        }
        const range = sel.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        if (!rect || (!rect.width && !rect.height)) {
          this.hideSelectionPin();
          return;
        }
        this.selPin.style.top  = (rect.bottom + window.scrollY + 4) + "px";
        this.selPin.style.left = Math.max(8, rect.left + window.scrollX) + "px";
        this.selPin.classList.add("cp-show");
      });
      document.addEventListener("mousedown", (e) => {
        if (e.target !== this.selPin) this.hideSelectionPin();
      });

      // Inject 📌 pin buttons on every card tagged with data-ref-kind.
      this._injectPins();
    }

    _injectPins() {
      const hosts = document.querySelectorAll("[data-ref-kind]");
      hosts.forEach((host) => {
        if (host.dataset.refPinInjected === "1") return;
        host.dataset.refPinInjected = "1";
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "cp-pin";
        btn.title = "📌 发给助手";
        btn.textContent = "📌 引用";
        const heading = host.querySelector(":scope > header, :scope > h1, :scope > h2, :scope > h3, :scope > figcaption");
        if (heading) heading.appendChild(btn);
        else host.appendChild(btn);
      });
    }

    hideSelectionPin() {
      this.selPin.classList.remove("cp-show");
    }

    toggle() {
      if (this.panel.classList.contains("cp-open")) this.close();
      else this.open();
    }
    open()  {
      this.panel.classList.add("cp-open");
      this.fab.setAttribute("aria-expanded", "true");
      this.elTextarea.focus();
      this._persistState();
    }
    close() {
      this.panel.classList.remove("cp-open");
      this.fab.setAttribute("aria-expanded", "false");
      this._persistState();
    }
    _renderWebToggle() {
      this.elWebToggle.classList.toggle("cp-web-on", this.useWeb);
      this.elWebToggle.setAttribute("aria-pressed", this.useWeb ? "true" : "false");
      this.elWebToggle.title = this.useWeb ? "联网补充已开启" : "开启联网补充";
    }

    // ---- Chips ----
    addChipFromHost(host) {
      const kind = host.dataset.refKind;
      const id   = host.dataset.refId;
      const text = host.dataset.refText || "";
      if (kind === "chapter") {
        this.addChip({ kind: "chapter", id: parseInt(id, 10), label: `章节 ${id}`, text: hostPreviewText(host) });
      } else if (kind === "frame") {
        const img = host.querySelector("img");
        this.addChip({
          kind: "frame",
          id: parseInt(id, 10),
          label: `关键帧 F${id}`,
          text: hostPreviewText(host),
          image: img ? (img.currentSrc || img.src) : "",
        });
      } else if (kind === "selection") {
        const t = text || host.textContent.trim().slice(0, 80);
        this.addChip({ kind: "selection", text: t, label: "「" + t.slice(0, 24) + "」" });
      }
    }
    addChip(ref) {
      // Dedupe by (kind, id/text).
      const key = ref.kind + "|" + (ref.id != null ? ref.id : (ref.text || ""));
      if (this.chips.some((c) => c._key === key)) return;
      ref._key = key;
      this.chips.push(ref);
      this._renderChips();
      this.open();
    }
    removeChip(key) {
      this.chips = this.chips.filter((c) => c._key !== key);
      this._renderChips();
    }
    _renderChips() {
      this.elChips.innerHTML = "";
      this.elChips.classList.toggle("cp-chips-empty", !this.chips.length);
      if (!this.chips.length) {
        this.elChips.innerHTML = '<span class="cp-ref-empty">可拖选正文或点击卡片上的“📌 引用”添加上下文</span>';
        return;
      }
      this.chips.forEach((c) => {
        const el = document.createElement("article");
        el.className = "cp-chip";
        el.innerHTML = `
          ${c.image ? '<img class="cp-chip-thumb" alt="" />' : ""}
          <div class="cp-chip-body">
            <div class="cp-chip-kicker"></div>
            <div class="cp-chip-text"></div>
          </div>
          <button type="button" class="cp-chip-close" aria-label="移除">×</button>
        `;
        const thumb = el.querySelector(".cp-chip-thumb");
        if (thumb) thumb.src = c.image;
        el.querySelector(".cp-chip-kicker").textContent = `${refKindLabel(c.kind)} · ${c.label || ""}`;
        el.querySelector(".cp-chip-text").textContent = refSummary(c);
        el.querySelector(".cp-chip-close").addEventListener("click", () => this.removeChip(c._key));
        this.elChips.appendChild(el);
      });
    }

    // ---- Messages ----
    _appendUserMessage(text, refs = []) {
      const el = document.createElement("div");
      el.className = "cp-message cp-message-user";
      const question = document.createElement("div");
      question.className = "cp-user-question";
      question.textContent = text;
      el.appendChild(question);
      if (refs.length) {
        const wrap = document.createElement("div");
        wrap.className = "cp-user-refs";
        const label = document.createElement("div");
        label.className = "cp-user-refs-label";
        label.textContent = "引用：";
        wrap.appendChild(label);
        refs.forEach((ref) => {
          const chip = document.createElement("article");
          chip.className = "cp-user-ref";
          chip.innerHTML = `
            ${ref.image ? '<img class="cp-user-ref-thumb" alt="" />' : ""}
            <div>
              <div class="cp-user-ref-kicker"></div>
              <div class="cp-user-ref-text"></div>
            </div>
          `;
          const thumb = chip.querySelector(".cp-user-ref-thumb");
          if (thumb) thumb.src = ref.image;
          chip.querySelector(".cp-user-ref-kicker").textContent = `${refKindLabel(ref.kind)} · ${ref.label || ""}`;
          chip.querySelector(".cp-user-ref-text").textContent = refSummary(ref);
          wrap.appendChild(chip);
        });
        el.appendChild(wrap);
      }
      this.elMessages.appendChild(el);
      this.elMessages.scrollTop = this.elMessages.scrollHeight;
    }
    _startAssistantMessage() {
      const el = document.createElement("div");
      el.className = "cp-message cp-message-assistant";
      el.innerHTML = `<div class="cp-answer"></div>` +
                     `<div class="cp-tool-trace" data-role="trace"></div>`;
      this.elMessages.appendChild(el);
      this.elMessages.scrollTop = this.elMessages.scrollHeight;
      this.currentAssistantEl = el.querySelector(".cp-answer");
      this.currentTraceEl     = el.querySelector("[data-role=trace]");
      this.currentAssistantRaw = "";
    }
    _appendAssistantMessage(text) {
      const el = document.createElement("div");
      el.className = "cp-message cp-message-assistant";
      el.innerHTML = `<div class="cp-answer"></div>`;
      el.querySelector(".cp-answer").innerHTML = renderMarkdown(text);
      this.elMessages.appendChild(el);
      this.elMessages.scrollTop = this.elMessages.scrollHeight;
    }
    _appendToken(delta) {
      if (!this.currentAssistantEl) this._startAssistantMessage();
      this.currentAssistantRaw += delta;
      this.currentAssistantEl.innerHTML = renderMarkdown(this.currentAssistantRaw);
      this.elMessages.scrollTop = this.elMessages.scrollHeight;
    }
    _appendToolCall(name, args) {
      if (!this.currentTraceEl) return;
      const line = document.createElement("span");
      line.className = "cp-tool-trace-item";
      const argKeys = args && typeof args === "object" ? Object.keys(args).slice(0, 3).join(",") : "";
      line.textContent = "→ " + name + (argKeys ? ` (${argKeys})` : "");
      this.currentTraceEl.appendChild(line);
    }
    _appendToolResult(payload) {
      if (!this.currentTraceEl) return;
      const line = document.createElement("span");
      line.className = "cp-tool-trace-item";
      if (payload && payload.ok === false) {
        line.textContent = "✖ " + (payload.name || "") + (payload.error && payload.error.code ? ` · ${payload.error.code}` : "");
      } else {
        const bits = [];
        if (payload && typeof payload.chunks_count === "number") bits.push(`${payload.chunks_count} 片段`);
        if (payload && typeof payload.chapter_idx === "number")  bits.push(`Ch${payload.chapter_idx}`);
        if (payload && typeof payload.frame_id === "number")     bits.push(`F${payload.frame_id}`);
        line.textContent = "✓ " + (payload.name || "") + (bits.length ? ` · ${bits.join(", ")}` : "");
      }
      this.currentTraceEl.appendChild(line);
    }
    _applyDone(payload) {
      if (payload && payload.patched_answer) {
        this.currentAssistantRaw = payload.patched_answer;
        this.currentAssistantEl.innerHTML = renderMarkdown(this.currentAssistantRaw);
      }
      if (payload && payload.warnings && payload.warnings.length) {
        const warnBox = document.createElement("div");
        warnBox.className = "cp-warnings";
        warnBox.textContent =
          "⚠ 检测到 " + payload.warnings.length +
          " 个越界锚点，已在文中标注（" +
          payload.warnings.map((w) => `${w.kind}:${w.value}`).join("，") + "）";
        this.currentAssistantEl.parentNode.appendChild(warnBox);
      }
    }
    _appendErrorMessage(msg) {
      if (this.currentAssistantEl && !this.currentAssistantRaw) {
        const emptyAssistant = this.currentAssistantEl.closest(".cp-message-assistant");
        if (emptyAssistant) emptyAssistant.remove();
        this.currentAssistantEl = null;
        this.currentTraceEl = null;
      }
      const box = document.createElement("div");
      box.className = "cp-message cp-message-error";
      const title = document.createElement("strong");
      title.textContent = "这次没有生成回答";
      const detail = document.createElement("div");
      detail.className = "cp-error-detail";
      detail.textContent = friendlyErrorMessage(msg);
      box.appendChild(title);
      box.appendChild(detail);
      this.elMessages.appendChild(box);
      this.elMessages.scrollTop = this.elMessages.scrollHeight;
    }

    _persistState() {
      const state = {
        version: STATE_VERSION,
        history: this.history.map(storedHistoryTurn).filter(Boolean).slice(-(MAX_HISTORY_TURNS * 2)),
        draft: this.elTextarea ? this.elTextarea.value : "",
        panel_open: this.panel ? this.panel.classList.contains("cp-open") : false,
      };
      safeStorageSet(this.storageKey, JSON.stringify(state));
    }

    _restoreState() {
      const raw = safeStorageGet(this.storageKey);
      if (!raw) return;
      let state;
      try { state = JSON.parse(raw); } catch { return; }
      const turns = Array.isArray(state.history)
        ? state.history.map(storedHistoryTurn).filter(Boolean).slice(-(MAX_HISTORY_TURNS * 2))
        : [];
      this.history = turns;
      this.elMessages.innerHTML = "";
      for (const turn of turns) {
        if (turn.role === "user") this._appendUserMessage(turn.content, turn.refs || []);
        else if (turn.role === "assistant") this._appendAssistantMessage(turn.content);
      }
      if (typeof state.draft === "string") this.elTextarea.value = state.draft;
      if (state.panel_open) {
        this.panel.classList.add("cp-open");
        this.fab.setAttribute("aria-expanded", "true");
      }
      if (turns.length) {
        this.elStatus.textContent = "已恢复上次对话";
        setTimeout(() => {
          if (this.elStatus.textContent === "已恢复上次对话") this.elStatus.textContent = "";
        }, 1800);
      }
    }

    clearConversation() {
      if (this.history.length || this.elMessages.children.length || this.elTextarea.value.trim()) {
        const ok = window.confirm ? window.confirm("清空当前讲义的对话历史？") : true;
        if (!ok) return;
      }
      if (this.abortController) this.abortController.abort();
      this.history = [];
      this.currentAssistantEl = null;
      this.currentTraceEl = null;
      this.currentAssistantRaw = "";
      this.elMessages.innerHTML = "";
      this.elTextarea.value = "";
      safeStorageRemove(this.storageKey);
      this._persistState();
      this.elStatus.textContent = "已清空当前讲义对话";
      setTimeout(() => {
        if (this.elStatus.textContent === "已清空当前讲义对话") this.elStatus.textContent = "";
      }, 1800);
      this.elTextarea.focus();
    }

    // ---- Send ----
    async send() {
      const question = this.elTextarea.value.trim();
      if (!question) return;
      if (this.abortController) this.abortController.abort();

      const usedChips = this.chips.slice();
      this._appendUserMessage(question, usedChips);
      this._startAssistantMessage();
      this.elTextarea.value = "";
      this.elTextarea.focus();
      this.elSend.disabled = true;
      this.elStatus.textContent = "正在思考…";

      const body = {
        bv: this.bv,
        question,
        references: this.chips.map((c) => ({
          kind: c.kind, id: c.id, text: c.text || null,
        })),
        use_web: this.useWeb,
        messages: this.history.slice(-MAX_HISTORY_TURNS).map((m) => ({
          role: m.role,
          content: m.content,
        })),
      };

      // After building the request, clear chips — treat them as
      // single-shot hints (matches the "三种引用" UX).
      this.chips = [];
      this._renderChips();
      this._persistState();

      this.abortController = new AbortController();
      try {
        for await (const ev of streamAsk(body, this.abortController.signal)) {
          if (ev.event === "token") this._appendToken(ev.data.delta || "");
          else if (ev.event === "tool_call") this._appendToolCall(ev.data.name || "", ev.data.args || {});
          else if (ev.event === "tool_result") this._appendToolResult(ev.data || {});
          else if (ev.event === "done") this._applyDone(ev.data || {});
          else if (ev.event === "error") this._appendErrorMessage((ev.data && ev.data.message) || "unknown");
        }
        const answer = this.currentAssistantRaw;
        this.history.push({
          role: "user",
          content: question,
          refs: usedChips.map(storedRef).filter(Boolean),
        });
        if (answer) this.history.push({ role: "assistant", content: answer });
        if (this.history.length > MAX_HISTORY_TURNS * 2) {
          this.history = this.history.slice(-MAX_HISTORY_TURNS * 2);
        }
        this._persistState();
      } catch (err) {
        if (err.name !== "AbortError") {
          this._appendErrorMessage(err.message || String(err));
        }
      } finally {
        this.elSend.disabled = false;
        this.elStatus.textContent = "";
        this.abortController = null;
      }
    }
  }

  // ------------------------------------------------------------------
  // Bootstrap
  // ------------------------------------------------------------------

  function init() {
    const stub = document.getElementById("cp-panel");
    ensureLatestCopilotCss();
    initImageLightbox();
    if (!stub) return;
    window.__copilot = new CopilotPanel(stub);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Expose for tests / debugging.
  window.LectureMindCopilot = {
    replaceAnchors,
    parseSseChunk,
    renderMarkdown,
    transformSections,
    timestampToSeconds,
    findAnchorTarget,
  };
})();
