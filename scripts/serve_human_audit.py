#!/usr/bin/env python3
"""Serve a localhost-only, two-stage private human-audit interface."""

from __future__ import annotations

import argparse
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

from topocf_rag.human_audit import (
    HumanAuditBundle,
    HumanAuditInvariantError,
    build_annotation_state,
    build_item_payload,
    load_annotation_events,
    load_human_audit_bundle,
    record_blind_decision,
    record_final_decision,
    write_public_human_audit_report,
)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TopoCF-RAG 私有人工核验</title>
  <style>
    :root { color-scheme: light; --ink:#17202a; --muted:#5d6d7e;
      --line:#d5d8dc; --bg:#f4f6f7; --card:#fff; --accent:#1f618d;
      --warn:#9a7d0a; --ok:#196f3d; --danger:#922b21; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink);
      font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    main { max-width:1080px; margin:0 auto; padding:24px; }
    header { position:sticky; top:0; z-index:2; background:rgba(244,246,247,.96);
      padding:12px 0; border-bottom:1px solid var(--line); }
    h1 { font-size:22px; margin:0 0 8px; }
    h2 { font-size:18px; margin:0 0 10px; }
    h3 { font-size:15px; margin:14px 0 6px; }
    .muted { color:var(--muted); }
    .card { background:var(--card); border:1px solid var(--line);
      border-radius:10px; padding:18px; margin:16px 0; box-shadow:0 1px 2px #0000000a; }
    .document { border-left:4px solid #85c1e9; padding:10px 14px; margin:12px 0;
      background:#f8fbfd; }
    .fact { padding:9px 12px; margin:8px 0; border-radius:6px; background:#fafafa; }
    .fact.missing { border-left:4px solid #e6b0aa; }
    .fact.retrieved { border-left:4px solid #82e0aa; }
    .badge { display:inline-block; border-radius:999px; padding:2px 9px;
      font-size:12px; margin-right:8px; background:#eaecee; }
    .badge.ok { background:#d5f5e3; color:var(--ok); }
    .badge.warn { background:#fcf3cf; color:var(--warn); }
    .nav { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
    button { border:0; border-radius:7px; padding:9px 14px; cursor:pointer;
      color:#fff; background:var(--accent); font-weight:600; }
    button.secondary { background:#566573; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    label.option { display:block; padding:10px; margin:8px 0; border:1px solid var(--line);
      border-radius:7px; cursor:pointer; }
    label.option:hover { background:#f8f9f9; }
    label.option code { font-size:12px; overflow-wrap:anywhere; }
    textarea { width:100%; min-height:76px; resize:vertical; padding:9px;
      border:1px solid var(--line); border-radius:7px; font:inherit; }
    .locked { border-left:4px solid var(--ok); background:#eafaf1; }
    .model { border-left:4px solid var(--warn); background:#fef9e7; }
    .error { color:var(--danger); white-space:pre-wrap; }
    .status { color:var(--ok); min-height:1.5em; }
    ul { margin:6px 0; padding-left:24px; }
    code { overflow-wrap:anywhere; }
  </style>
</head>
<body>
<main>
  <header>
    <h1>TopoCF-RAG 私有人工核验</h1>
    <div id="progress" class="muted">加载中…</div>
    <div class="nav">
      <button id="previous" class="secondary">上一条</button>
      <button id="next" class="secondary">下一条</button>
      <button id="next-incomplete">下一条未完成</button>
    </div>
  </header>
  <div id="error" class="error"></div>
  <div id="status" class="status"></div>
  <section id="question" class="card"></section>
  <section id="documents" class="card"></section>
  <section id="gold" class="card"></section>
  <section id="blind" class="card"></section>
  <section id="model" class="card model"></section>
  <section id="final" class="card"></section>
</main>
<script>
"use strict";
const labelOrder = [
  "locally_true_globally_incomplete",
  "complete_via_alternative_proof",
  "missing_all_gold_evidence",
  "ambiguous_or_annotation_issue",
  "other"
];
const params = new URLSearchParams(window.location.search);
let token = params.get("token") || sessionStorage.getItem("topocfAuditToken");
if (params.get("token")) {
  sessionStorage.setItem("topocfAuditToken", params.get("token"));
  history.replaceState({}, "", window.location.pathname);
}
let currentIndex = 0;
let current = null;

function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined && text !== null) result.textContent = String(text);
  if (className) result.className = className;
  return result;
}
function clear(element) { element.replaceChildren(); }
function setError(message) { document.getElementById("error").textContent = message || ""; }
function setStatus(message) { document.getElementById("status").textContent = message || ""; }

async function api(path, options = {}) {
  if (!token) throw new Error("URL 中缺少核验 token；请使用服务器输出的完整地址。");
  const headers = Object.assign({}, options.headers || {}, {"X-Audit-Token": token});
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, Object.assign({}, options, {headers, cache:"no-store"}));
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function addLabelOptions(container, groupName, selected, definitions) {
  for (const label of labelOrder) {
    const wrapper = node("label", null, "option");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = groupName;
    input.value = label;
    input.checked = label === selected;
    wrapper.append(input, document.createTextNode(" "));
    wrapper.append(node("code", label));
    wrapper.append(node("div", definitions[label], "muted"));
    container.append(wrapper);
  }
}

function selectedLabel(groupName) {
  const selected = document.querySelector(`input[name="${groupName}"]:checked`);
  return selected ? selected.value : null;
}

function renderQuestion(data) {
  const box = document.getElementById("question"); clear(box);
  box.append(node("h2", "问题与参考答案"));
  box.append(node("h3", "Question"), node("div", data.question));
  box.append(node("h3", "Reference answer"), node("div", data.reference_answer));
}

function renderDocuments(data) {
  const box = document.getElementById("documents"); clear(box);
  box.append(node("h2", "实际检索到的文档"));
  data.retrieved_documents.forEach((doc, index) => {
    const item = node("div", null, "document");
    const rank = doc.retrieval_rank === undefined ? index + 1 : doc.retrieval_rank;
    item.append(node("h3", `Rank ${rank}: ${doc.title}`));
    if (doc.retrieval_score !== undefined) {
      item.append(node("div", `retrieval score: ${doc.retrieval_score}`, "muted"));
    }
    const list = document.createElement("ul");
    doc.sentences.forEach(sentence => list.append(node("li", sentence)));
    item.append(list); box.append(item);
  });
}

function renderGold(data) {
  const box = document.getElementById("gold"); clear(box);
  box.append(node("h2", "Gold supporting facts（仅用于裁决）"));
  box.append(node("div", "retrieved=false 的事实不属于当前可用证据；请同时检查检索文档是否存在替代证明。", "muted"));
  data.gold_supporting_facts_for_adjudication.forEach(fact => {
    const item = node("div", null, fact.retrieved ? "fact retrieved" : "fact missing");
    item.append(node("span", fact.retrieved ? "已检索" : "未检索", `badge ${fact.retrieved ? "ok" : "warn"}`));
    item.append(node("strong", fact.title));
    item.append(node("div", fact.sentence));
    box.append(item);
  });
}

function renderBlind(data) {
  const box = document.getElementById("blind"); clear(box);
  box.append(node("h2", "阶段 1：独立盲判"));
  if (data.blind_decision) {
    box.classList.add("locked");
    box.append(node("div", "盲判已经锁定，不能修改。"));
    box.append(node("code", data.blind_decision.label));
    if (data.blind_decision.notes) box.append(node("div", data.blind_decision.notes, "muted"));
    return;
  }
  box.classList.remove("locked");
  box.append(node("div", "此阶段看不到 Qwen 结果。请只根据上面的证据选择标签。", "muted"));
  const options = node("div"); addLabelOptions(options, "blind-label", null, data.label_definitions); box.append(options);
  const notes = document.createElement("textarea"); notes.id = "blind-notes";
  notes.placeholder = "可选：记录关键依据（最多 2000 字符）"; notes.maxLength = 2000; box.append(notes);
  const button = node("button", "锁定盲判并揭示 Qwen");
  button.addEventListener("click", async () => {
    const label = selectedLabel("blind-label");
    if (!label) { setError("请先选择盲判标签。"); return; }
    if (!confirm("盲判保存后不可修改。确认锁定？")) return;
    await submitDecision("/api/blind", {index:currentIndex, label, notes:notes.value}, "盲判已锁定。Qwen 建议现在可见。");
  });
  box.append(button);
}

function renderModel(data) {
  const box = document.getElementById("model"); clear(box);
  box.append(node("h2", "阶段 2：Qwen 预标注（盲判后揭示）"));
  if (!data.prelabel) {
    box.hidden = true; return;
  }
  box.hidden = false;
  const judgment = data.prelabel.judgment;
  box.append(node("div", `status: ${data.prelabel.status}`));
  if (!judgment) {
    box.append(node("div", `validation errors: ${(data.prelabel.validation_errors || []).join("; ")}`, "error"));
    return;
  }
  box.append(node("h3", "Qwen label"), node("code", judgment.label));
  box.append(node("div", `confidence: ${judgment.confidence}`));
  box.append(node("h3", "Rationale"), node("div", judgment.rationale));
  box.append(node("h3", "Missing requirement"), node("div", judgment.missing_requirement || "(empty)"));
}

function renderFinal(data) {
  const box = document.getElementById("final"); clear(box);
  box.append(node("h2", "阶段 3：最终人工裁决"));
  if (!data.blind_decision) {
    box.hidden = true; return;
  }
  box.hidden = false;
  box.append(node("div", "看到 Qwen 建议后，确认或修正最终标签。最终标签可以再次修改。", "muted"));
  const selected = data.final_decision ? data.final_decision.label : data.blind_decision.label;
  const options = node("div"); addLabelOptions(options, "final-label", selected, data.label_definitions); box.append(options);
  const notes = document.createElement("textarea"); notes.id = "final-notes";
  notes.placeholder = "可选：记录最终裁决依据（最多 2000 字符）"; notes.maxLength = 2000;
  notes.value = data.final_decision ? data.final_decision.notes : ""; box.append(notes);
  const button = node("button", data.final_decision ? "更新最终裁决" : "保存最终裁决");
  button.addEventListener("click", async () => {
    const label = selectedLabel("final-label");
    await submitDecision("/api/final", {index:currentIndex, label, notes:notes.value}, "最终裁决已保存。");
  });
  box.append(button);
}

function render(data) {
  current = data; currentIndex = data.index;
  const p = data.progress;
  document.getElementById("progress").textContent = `第 ${data.index + 1}/${data.total} 条 · 盲判 ${p.blind_completed}/${p.total} · 最终完成 ${p.final_completed}/${p.total}`;
  document.getElementById("previous").disabled = data.index === 0;
  document.getElementById("next").disabled = data.index + 1 >= data.total;
  document.getElementById("next-incomplete").disabled = p.next_incomplete_index === null;
  renderQuestion(data); renderDocuments(data); renderGold(data); renderBlind(data); renderModel(data); renderFinal(data);
}

async function loadItem(index) {
  setError(""); setStatus("");
  try { render(await api(`/api/item?index=${index}`)); window.scrollTo({top:0, behavior:"smooth"}); }
  catch (error) { setError(error.message); }
}

async function submitDecision(path, payload, message) {
  setError(""); setStatus("");
  try {
    render(await api(path, {method:"POST", body:JSON.stringify(payload)}));
    setStatus(message);
  } catch (error) { setError(error.message); }
}

document.getElementById("previous").addEventListener("click", () => loadItem(currentIndex - 1));
document.getElementById("next").addEventListener("click", () => loadItem(currentIndex + 1));
document.getElementById("next-incomplete").addEventListener("click", () => {
  if (current && current.progress.next_incomplete_index !== null) loadItem(current.progress.next_incomplete_index);
});
loadItem(0);
</script>
</body>
</html>
"""


class AuditApplication:
    def __init__(
        self,
        bundle: HumanAuditBundle,
        annotations_path: Path,
        report_path: Path,
    ) -> None:
        self.bundle = bundle
        self.annotations_path = annotations_path
        self.report_path = report_path
        self.lock = threading.Lock()
        write_public_human_audit_report(bundle, annotations_path, report_path)

    def _state(self) -> dict[str, dict[str, Any]]:
        events = load_annotation_events(self.annotations_path, self.bundle)
        return build_annotation_state(events, self.bundle)

    def item(self, index: int) -> dict[str, Any]:
        with self.lock:
            return build_item_payload(self.bundle, self._state(), index)

    def decide(
        self, event_type: str, *, index: int, label: str, notes: str
    ) -> dict[str, Any]:
        if not isinstance(index, int) or isinstance(index, bool):
            raise HumanAuditInvariantError("item index must be an integer")
        if not 0 <= index < len(self.bundle.records):
            raise HumanAuditInvariantError("item index is out of range")
        qid = self.bundle.question_ids[index]
        with self.lock:
            if event_type == "blind_decision":
                record_blind_decision(
                    self.bundle,
                    self.annotations_path,
                    question_id=qid,
                    label=label,
                    notes=notes,
                )
            elif event_type == "final_decision":
                record_final_decision(
                    self.bundle,
                    self.annotations_path,
                    question_id=qid,
                    label=label,
                    notes=notes,
                )
            else:
                raise HumanAuditInvariantError("unsupported decision type")
            write_public_human_audit_report(
                self.bundle, self.annotations_path, self.report_path
            )
            return build_item_payload(self.bundle, self._state(), index)


def handler_class(application: AuditApplication, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "TopoCFPrivateAudit/1"

        def log_message(self, _format: str, *args: Any) -> None:
            # Suppress paths so the one-time token is never written to logs.
            return

        def _security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; connect-src 'self'; "
                "img-src 'none'; frame-ancestors 'none'; base-uri 'none'",
            )

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self._security_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _authorized(self) -> bool:
            supplied = self.headers.get("X-Audit-Token", "")
            return bool(supplied) and compare_digest(supplied, token)

        def _require_authorized(self) -> bool:
            if self._authorized():
                return True
            self._json(403, {"error": "invalid or missing audit token"})
            return False

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if not self._require_authorized():
                return
            try:
                if parsed.path == "/api/item":
                    values = parse_qs(parsed.query).get("index", [])
                    if len(values) != 1:
                        raise HumanAuditInvariantError("exactly one index is required")
                    index = int(values[0])
                    self._json(200, application.item(index))
                else:
                    self._json(404, {"error": "not found"})
            except (HumanAuditInvariantError, ValueError) as error:
                self._json(400, {"error": str(error)})

        def do_POST(self) -> None:  # noqa: N802
            if not self._require_authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024:
                    raise HumanAuditInvariantError("request body size is invalid")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise HumanAuditInvariantError("request body must be an object")
                index = payload.get("index")
                label = payload.get("label")
                notes = payload.get("notes", "")
                path = urlparse(self.path).path
                if path == "/api/blind":
                    result = application.decide(
                        "blind_decision", index=index, label=label, notes=notes
                    )
                elif path == "/api/final":
                    result = application.decide(
                        "final_decision", index=index, label=label, notes=notes
                    )
                else:
                    self._json(404, {"error": "not found"})
                    return
                self._json(200, result)
            except (HumanAuditInvariantError, ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prelabels", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    input_path = args.input.resolve()
    prelabels_path = args.prelabels.resolve()
    annotations_path = args.annotations.resolve()
    for name, path in (
        ("private input", input_path),
        ("private prelabels", prelabels_path),
        ("private annotations", annotations_path),
    ):
        if path.is_relative_to(repository_root):
            raise HumanAuditInvariantError(
                f"{name} must be outside the Git repository"
            )
    if not 1 <= args.port <= 65535:
        raise HumanAuditInvariantError("port must be between 1 and 65535")
    bundle = load_human_audit_bundle(input_path, prelabels_path)
    application = AuditApplication(bundle, annotations_path, args.report.resolve())
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler_class(application, token)
    )
    print("private_audit_bind=127.0.0.1", flush=True)
    print(f"private_audit_port={args.port}", flush=True)
    print(
        f"private_audit_url=http://127.0.0.1:{args.port}/?token={token}",
        flush=True,
    )
    print("Press Ctrl-C to stop; progress is append-only and resumable.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping private audit server.", flush=True)
    finally:
        server.server_close()
        write_public_human_audit_report(
            bundle, annotations_path, args.report.resolve()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
