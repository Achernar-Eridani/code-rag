import * as vscode from "vscode";
import { ping, search, explain, friendlyError, SearchResponse } from "./api";
import { SearchTreeProvider, registerSearchOpenCommand } from "./searchView";
import { ExplainPanel } from "./explainPanel";

export async function activate(context: vscode.ExtensionContext) {
  // 状态栏（插件通过命令激活后再加载，不阻塞启动）
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  status.text = "RAG: ready";
  status.show();
  context.subscriptions.push(status);

  // TreeView（已在 package.json 声明 ragSearchView）
  const tree = new SearchTreeProvider();
  const treeView = vscode.window.createTreeView("ragSearchView", { treeDataProvider: tree });
  context.subscriptions.push(treeView);
  registerSearchOpenCommand(context);

  // 首次调用前 ping
  const info = await ping();
  status.text = info.ok ? `RAG: ${info.provider} · ${info.model}` : "RAG: offline";

  // RAG: Search Code
  context.subscriptions.push(vscode.commands.registerCommand("rag.search", async () => {
    const editor = vscode.window.activeTextEditor;
    const sel = editor?.document.getText(editor.selection) || "";
    const q = await vscode.window.showInputBox({ title: "RAG: Search Code", value: sel, prompt: "Enter your search query" });
    if (!q) return;

    try {
      status.text = "RAG: searching…";
      const res: SearchResponse = await search(q);
      tree.refresh(res.results || []);
      vscode.window.showInformationMessage(`Found ${res.total} results.`);
    } catch (e: any) {
      vscode.window.showErrorMessage(`Search failed: ${friendlyError(e)}`);
    } finally {
      const info2 = await ping();
      status.text = info2.ok ? `RAG: ${info2.provider} · ${info2.model}` : "RAG: offline";
    }
  }));

  // RAG: Explain Selection
  context.subscriptions.push(vscode.commands.registerCommand("rag.explain", async () => {
    const editor = vscode.window.activeTextEditor;
    const sel = editor?.document.getText(editor.selection) || "";
    const q = await vscode.window.showInputBox({ title: "RAG: Explain Selection", value: sel, prompt: "Enter a question about current code" });
    if (!q) return;

    try {
      status.text = "RAG: explaining…";
      const resp = await explain(q);
      ExplainPanel.show(context, resp);
    } catch (e: any) {
      vscode.window.showErrorMessage(`Explain failed: ${friendlyError(e)}`);
    } finally {
      const info2 = await ping();
      status.text = info2.ok ? `RAG: ${info2.provider} · ${info2.model}` : "RAG: offline";
    }
  }));
}

export function deactivate() {}
