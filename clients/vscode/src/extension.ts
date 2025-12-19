import * as vscode from "vscode";
import { ping, search, explain, agentExplain, friendlyError, SearchResponse } from "./api";
import { SearchTreeProvider, registerSearchOpenCommand } from "./searchView";
import { ExplainPanel } from "./explainPanel";
import { buildAndUploadIndex } from "./indexer"; // 引用新模块

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

  // 3. 监听配置变化
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("rag")) {
        refreshStatusBar();
      }
    })
  );

  refreshStatusBar();

  // --- Command: Search ---
  context.subscriptions.push(vscode.commands.registerCommand("rag.search", async () => {
    const editor = vscode.window.activeTextEditor;
    const sel = editor?.document.getText(editor.selection) || "";
    const q = await vscode.window.showInputBox({ title: "RAG: Search Code", value: sel, prompt: "Enter your search query" });
    if (!q) return;

    try {
      statusBarItem.text = "$(sync~spin) RAG: searching..."; 
      const res: SearchResponse = await search(q);
      tree.refresh(res.results || []);
      vscode.window.showInformationMessage(`Found ${res.total} results.`);
    } catch (e: any) {
      vscode.window.showErrorMessage(`Search failed: ${friendlyError(e)}`);
    } finally {
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

  // --- Command: Agent ---
  context.subscriptions.push(
    vscode.commands.registerCommand("rag.agentExplain", async () => {
      const editor = vscode.window.activeTextEditor;
      const sel = editor?.document.getText(editor.selection) || "";

      const q = await vscode.window.showInputBox({
        title: "RAG Agent: Ask Code Agent",
        value: sel,
        prompt: "Ask anything about this repo. Agent will decide whether to search.",
      });
      if (!q) return;

      try {
        statusBarItem.text = "$(sync~spin) RAG: agent thinking...";
        const resp = await agentExplain(q);
        ExplainPanel.showAgent(context, resp);
      } catch (e: any) {
        vscode.window.showErrorMessage(`Code Agent failed: ${friendlyError(e)}`);
      } finally {
        await refreshStatusBar();
      }
    })
  );

  // --- Command: Build Index (New) ---
  context.subscriptions.push(
      vscode.commands.registerCommand("rag.buildIndex", async () => {
          try {
              statusBarItem.text = "$(sync~spin) RAG: Indexing...";
              await buildAndUploadIndex({ fresh: true });
              vscode.window.showInformationMessage("Workspace Index Rebuilt Successfully!");
          } catch (e: any) {
              vscode.window.showErrorMessage(`Indexing failed: ${e.message}`);
          } finally {
              await refreshStatusBar();
          }
      })
  );
}

export function deactivate() {}

async function refreshStatusBar() {
  try {
    const info = await ping();
    if (info.ok) {
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