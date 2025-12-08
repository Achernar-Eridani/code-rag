import * as vscode from "vscode";
import { ExplainResponse, AgentExplainResponse, AgentToolResult } from "./api";

export class ExplainPanel {
  public static current: ExplainPanel | null = null;
  private panel: vscode.WebviewPanel;

  // ===========================================================================
  //  Static Methods (Entry Points)
  // ===========================================================================

  /**
   * 模式 1：普通 Explain 展示
   */
  static show(ctx: vscode.ExtensionContext, resp: ExplainResponse) {
    if (ExplainPanel.current) {
      ExplainPanel.current.renderExplain(resp);
      ExplainPanel.current.panel.reveal(vscode.ViewColumn.Beside);
      return;
    }
    const panel = ExplainPanel.createPanel(ctx);
    ExplainPanel.current = new ExplainPanel(panel, ctx);
    ExplainPanel.current.renderExplain(resp);
  }

  /**
   * 模式 2：Agent 展示 (新增)
   */
  static showAgent(ctx: vscode.ExtensionContext, resp: AgentExplainResponse) {
    if (ExplainPanel.current) {
      ExplainPanel.current.renderAgent(resp);
      ExplainPanel.current.panel.reveal(vscode.ViewColumn.Beside);
      return;
    }
    const panel = ExplainPanel.createPanel(ctx);
    ExplainPanel.current = new ExplainPanel(panel, ctx);
    ExplainPanel.current.renderAgent(resp);
  }

  // ===========================================================================
  //  Private / Instance Methods
  // ===========================================================================

  private static createPanel(ctx: vscode.ExtensionContext): vscode.WebviewPanel {
    return vscode.window.createWebviewPanel(
      "ragExplain",
      "RAG Explain",
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true }
    );
  }

  // 构造函数现在只负责绑定事件，不负责初始渲染
  private constructor(panel: vscode.WebviewPanel, ctx: vscode.ExtensionContext) {
    this.panel = panel;
    this.panel.onDidDispose(() => {
      ExplainPanel.current = null;
    });

    // 统一处理前端发回来的 "open" 命令
    this.panel.webview.onDidReceiveMessage(async (msg) => {
      if (msg?.command === "open") {
        const { path, start, end } = msg;
        await vscode.commands.executeCommand(
          "rag.openSearchTarget",
          path,
          start,
          end
        );
      }
    });
  }

  /**
   * 渲染普通 Explain 结果 (原 update 方法改造)
   */
  private renderExplain(resp: ExplainResponse) {
    const warn = resp.answer.trim().startsWith("（降级：LLM 不可用）");
    const header = `**Provider**: \`${resp.provider}\` · **Model**: \`${
      resp.model
    }\` · **Latency**: retrieval ${
      resp.timings_ms?.retrieval ?? 0
    } ms / gen ${resp.timings_ms?.generation ?? 0} ms`;

    const evidences = (resp.evidences || [])
      .map((e, idx) => {
        const label = `[#${idx + 1}] ${e.kind} ${e.name} @ ${e.path} L${
          e.start_line
        }-${e.end_line}`;
        
        // 构造跳转数据 JSON
        const evt = { path: e.path, start: e.start_line, end: e.end_line };
        const evtJson = JSON.stringify(evt).replace(/"/g, "&quot;");
        
        return `<li>${escapeHtml(label)} <button data-evt="${evtJson}">Open</button></li>`;
      })
      .join("");

    const bodyHtml = `
      ${warn ? `<div class="warn">No LLM output – showing evidence summary returned by backend.</div>` : ``}
      <div class="muted">${header}</div>
      <h2>Answer</h2>
      <div class="markdown-body">${markdownToHtml(resp.answer)}</div>
      <h2>Evidences</h2>
      <ul>${evidences}</ul>
    `;

    this.panel.webview.html = this.getHtmlWrapper(bodyHtml);
  }

  /**
   * 渲染 Agent 结果 (新增)
   */
  private renderAgent(resp: AgentExplainResponse) {
    const headerParts: string[] = [];

    if (resp.used_tool) {
      headerParts.push(`Used tool: \`${resp.used_tool}\``);
    } else {
      headerParts.push("Used tool: none (pure LLM answer)");
    }

    // if (resp.tool_input) {
    //   headerParts.push("Tool input attached below");
    // }

    const header = headerParts.join(" · ");

    // Tool input 以 JSON 形式打印一份
    const toolInputHtml = resp.tool_input
      ? `<pre>${escapeHtml(JSON.stringify(resp.tool_input, null, 2))}</pre>`
      : `<div class="muted">No tool input.</div>`;

    // Tool results
    const toolResults = resp.tool_results ?? [];
    const toolResultsHtml =
      toolResults.length > 0
        ? toolResults
            .map((e: AgentToolResult, idx) => {
              const label = `[#${idx + 1}] ${e.kind ?? ""} ${e.symbol ?? ""} @ ${
                e.path ?? ""
              } L${e.start_line ?? 0}-${e.end_line ?? 0} (score=${e.score ?? "N/A"})`;

              const evt = {
                path: e.path,
                start: e.start_line,
                end: e.end_line,
              };
              const evtJson = JSON.stringify(evt).replace(/"/g, "&quot;");

              return `<li>${escapeHtml(label)} <button data-evt="${evtJson}">Open</button></li>`;
            })
            .join("")
        : `<li class="muted">No tool results.</li>`;

    const bodyHtml = `
      <div class="query-box">Agent Q: ${escapeHtml(resp.query)}</div>

      <div class="muted">${header}</div>

      <h2>Agent Answer</h2>
      <div class="markdown-body">${markdownToHtml(resp.answer)}</div>

      <h2>Tool Input</h2>
      ${toolInputHtml}

      <h2>Tool Results</h2>
      <ul>${toolResultsHtml}</ul>
    `;

    this.panel.webview.html = this.getHtmlWrapper(bodyHtml);
  }

  /**
   * 通用 HTML 包装器 (包含 CSS 和 全局点击监听)
   */
  private getHtmlWrapper(bodyContent: string): string {
    return `<!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8" />
        <style>
          body { font-family: -apple-system, Segoe UI, system-ui, sans-serif; padding: 12px; line-height: 1.5; }
          .warn { background: #fff3cd; border: 1px solid #ffeeba; padding: 8px 12px; margin-bottom: 10px; border-radius: 4px; }
          .muted { color: #666; font-size: 12px; }
          .query-box { background: #f0f0f0; padding: 8px; border-radius: 4px; margin-bottom: 12px; font-style: italic; color: #333; }
          h2 { margin: 16px 0 8px; font-size: 16px; border-bottom: 1px solid #eee; padding-bottom: 4px; }
          pre { background: #f6f8fa; padding: 8px; overflow: auto; border-radius: 4px; }
          code { font-family: Consolas, "Courier New", monospace; background: #f6f8fa; padding: 2px 4px; border-radius: 3px; }
          ul { padding-left: 18px; }
          li { margin-bottom: 4px; }
          button { margin-left: 8px; cursor: pointer; padding: 2px 6px; }
          
          /* 简单的 Markdown 样式 */
          .markdown-body p { margin-bottom: 8px; }
          .markdown-body ul { margin-bottom: 8px; }
        </style>
      </head>
      <body>
        ${bodyContent}

        <script>
          const vscode = acquireVsCodeApi();

          // 全局监听点击事件
          document.addEventListener("click", (e) => {
            if (e.target && e.target.tagName === "BUTTON") {
              try {
                // 读取按钮上的 JSON 数据
                const data = JSON.parse(e.target.getAttribute("data-evt"));
                // 发送给 extension
                vscode.postMessage({ command: "open", ...data });
              } catch (_) {}
            }
          });
        </script>
      </body>
      </html>`;
  }
}

// =============================================================================
//  Helpers
// =============================================================================

function escapeHtml(s: string): string {
  return (s || "").replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]!));
}

function markdownToHtml(md: string): string {
  let s = md || "";
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