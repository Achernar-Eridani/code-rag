import * as vscode from "vscode";
import { ping, search, explain, friendlyError, SearchResponse } from "./api";
import { SearchTreeProvider, registerSearchOpenCommand } from "./searchView";
import { ExplainPanel } from "./explainPanel";

let statusBarItem: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
  // 1. 初始化状态栏
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.text = "RAG: ready";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  // 2. 初始化 TreeView
  const tree = new SearchTreeProvider();
  const treeView = vscode.window.createTreeView("ragSearchView", { treeDataProvider: tree });
  context.subscriptions.push(treeView);
  registerSearchOpenCommand(context);

  // 3. 核心改进：监听配置变化
  // 当用户在设置里修改 rag.providerOverride 或 rag.apiKey 时，自动刷新状态栏
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("rag")) {
        refreshStatusBar();
      }
    })
  );

  // 4. 启动时刷新一次 (不要 await，避免阻塞插件激活)
  refreshStatusBar();

  // --- Command: Search ---
  context.subscriptions.push(vscode.commands.registerCommand("rag.search", async () => {
    const editor = vscode.window.activeTextEditor;
    const sel = editor?.document.getText(editor.selection) || "";
    const q = await vscode.window.showInputBox({ title: "RAG: Search Code", value: sel, prompt: "Enter your search query" });
    if (!q) return;

    try {
      statusBarItem.text = "$(sync~spin) RAG: searching..."; // 加个转圈图标
      const res: SearchResponse = await search(q);
      tree.refresh(res.results || []);
      vscode.window.showInformationMessage(`Found ${res.total} results.`);
    } catch (e: any) {
      vscode.window.showErrorMessage(`Search failed: ${friendlyError(e)}`);
    } finally {
      // 结束后恢复状态
      await refreshStatusBar();
    }
  }));

  // --- Command: Explain ---
  context.subscriptions.push(vscode.commands.registerCommand("rag.explain", async () => {
    const editor = vscode.window.activeTextEditor;
    const sel = editor?.document.getText(editor.selection) || "";
    const q = await vscode.window.showInputBox({ title: "RAG: Explain Selection", value: sel, prompt: "Enter a question about current code" });
    if (!q) return;

    try {
      statusBarItem.text = "$(sync~spin) RAG: explaining...";
      const resp = await explain(q);
      ExplainPanel.show(context, resp);
    } catch (e: any) {
      vscode.window.showErrorMessage(`Explain failed: ${friendlyError(e)}`);
    } finally {
      await refreshStatusBar();
    }
  }));
}

export function deactivate() {}

// 抽离出来的刷新逻辑
async function refreshStatusBar() {
  try {
    // 这里的 ping 内部会调用 buildHeaders() 读取最新配置
    const info = await ping();
    if (info.ok) {
      // 显示 Provider 和 Model
      statusBarItem.text = `$(check) RAG: ${info.provider} · ${info.model}`;
      statusBarItem.tooltip = "Code RAG Service is Online";
    } else {
      statusBarItem.text = "$(error) RAG: offline";
      statusBarItem.tooltip = "Cannot connect to RAG API";
    }
  } catch (e) {
    statusBarItem.text = "$(error) RAG: offline";
  }
}