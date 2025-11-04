import * as vscode from "vscode";
import { ExplainResponse } from "./api";

export class ExplainPanel {
  public static current: ExplainPanel | null = null;
  private panel: vscode.WebviewPanel;

  static show(ctx: vscode.ExtensionContext, resp: ExplainResponse) {
    if (ExplainPanel.current) {
      ExplainPanel.current.update(resp);
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      "ragExplain",
      "RAG Explain",
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true }
    );
    ExplainPanel.current = new ExplainPanel(panel, ctx, resp);
  }

  private constructor(panel: vscode.WebviewPanel, ctx: vscode.ExtensionContext, resp: ExplainResponse) {
    this.panel = panel;
    this.panel.onDidDispose(() => { ExplainPanel.current = null; });
    this.panel.webview.onDidReceiveMessage(async (msg) => {
      if (msg?.command === "open") {
        const { path, start, end } = msg;
        await vscode.commands.executeCommand("rag.openSearchTarget", path, start, end);
      }
    });
    this.update(resp);
  }

  update(resp: ExplainResponse) {
    const warn = resp.answer.trim().startsWith("（降级：LLM 不可用）");
    const header = `**Provider**: \`${resp.provider}\`  ·  **Model**: \`${resp.model}\`  ·  **Latency**: retrieval ${resp.timings_ms?.retrieval ?? 0} ms / gen ${resp.timings_ms?.generation ?? 0} ms`;

    const evidences = (resp.evidences || [])
      .map((e, idx) => {
        const label = `[#${idx + 1}] ${e.kind} ${e.name} @ ${e.path} L${e.start_line}-${e.end_line}`;
        return `<li>${escapeHtml(label)} <button data-idx="${idx}">Open</button></li>`;
      })
      .join("");

    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8" />
        <style>
          body { font-family: -apple-system, Segoe UI, system-ui, sans-serif; padding: 12px; }
          .warn { background: #fff3cd; border: 1px solid #ffeeba; padding: 8px 12px; margin-bottom: 10px; }
          .muted { color: #666; font-size: 12px; }
          h2 { margin: 8px 0 6px; }
          pre { background: #f6f8fa; padding: 8px; overflow: auto; }
          ul { padding-left: 18px; }
          button { margin-left: 8px; }
        </style>
      </head>
      <body>
        ${warn ? `<div class="warn">No LLM output – showing evidence summary returned by backend.</div>`: ``}
        <div class="muted">${header}</div>
        <h2>Answer</h2>
        <div>${markdownToHtml(resp.answer)}</div>
        <h2>Evidences</h2>
        <ul id="ev">${evidences}</ul>

        <script>
          const vscode = acquireVsCodeApi();
          document.getElementById("ev")?.addEventListener("click", (e) => {
            const t = e.target;
            if (t && t.tagName === "BUTTON") {
              const idx = t.getAttribute("data-idx");
              try {
                const data = ${JSON.stringify(resp.evidences || [])};
                const ev = data[Number(idx)];
                if (ev) vscode.postMessage({ command: "open", path: ev.path, start: ev.start_line, end: ev.end_line });
              } catch (_) {}
            }
          });
        </script>
      </body>
      </html>
    `;
    this.panel.webview.html = html;
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[m]!));
}

function markdownToHtml(md: string): string {
  let s = md;
  s = s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  s = s.replace(/```([\s\S]*?)```/g, (_m, code) => `<pre><code>${code}</code></pre>`);
  s = s.replace(/^### (.*)$/gm, "<h3>$1</h3>");
  s = s.replace(/^## (.*)$/gm, "<h2>$1</h2>");
  s = s.replace(/^# (.*)$/gm, "<h1>$1</h1>");
  s = s.replace(/^\- (.*)$/gm, "<li>$1</li>");
  s = s.replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>");
  s = s.replace(/\n{2,}/g, "<br/>");
  s = s.replace(/\n/g, "<br/>");
  return s;
}
