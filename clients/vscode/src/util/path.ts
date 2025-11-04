import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";

export function normalizePath(p: string): string {
  return p.replace(/\\/g, "/");
}

export function resolveInWorkspace(p: string): string | null {
  const cfg = vscode.workspace.getConfiguration("rag");
  const prefix = cfg.get<string>("repoPathPrefix") ?? "";

  const w = vscode.workspace.workspaceFolders?.[0];
  if (!w) return null;

  let rel = normalizePath(p);
  if (prefix && rel.startsWith(prefix)) {
    rel = rel.substring(prefix.length);
  }

  const abs = path.join(w.uri.fsPath, rel);
  if (fs.existsSync(abs)) return abs;

  // 再尝试去掉前导 ./ 等
  const alt = path.join(w.uri.fsPath, rel.replace(/^\.\//, ""));
  if (fs.existsSync(alt)) return alt;

  return null;
}

export async function openLocation(filePath: string, startLine?: number, endLine?: number) {
  const abs = resolveInWorkspace(filePath);
  if (!abs) {
    vscode.window.showWarningMessage(`File not found in workspace: ${filePath}`);
    return;
    }
  const doc = await vscode.workspace.openTextDocument(abs);
  const editor = await vscode.window.showTextDocument(doc, { preview: false });
  const start = new vscode.Position(Math.max((startLine ?? 1) - 1, 0), 0);
  const end = new vscode.Position(Math.max((endLine ?? startLine ?? 1) - 1, 0), 0);
  editor.revealRange(new vscode.Range(start, end), vscode.TextEditorRevealType.InCenter);
  editor.selection = new vscode.Selection(start, start);
}
