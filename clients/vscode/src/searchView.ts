import * as vscode from "vscode";
import { SearchItem } from "./api";
import { openLocation } from "./util/path";

export class SearchTreeProvider implements vscode.TreeDataProvider<SearchNode> {
  private _onDidChangeTreeData = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private items: SearchItem[] = [];

  refresh(results: SearchItem[]) {
    this.items = results ?? [];
    this._onDidChangeTreeData.fire();
  }

  clear() {
    this.items = [];
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: SearchNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: SearchNode): Thenable<SearchNode[]> {
    if (element) return Promise.resolve([]);
    return Promise.resolve(this.items.map(itemToNode));
  }
}

function itemToNode(it: SearchItem): SearchNode {
  const label = `${it.name || "(unnamed)"} (${it.kind || "?"})  ·  ${it.path}:L${it.start_line}-${it.end_line}  ·  ${it.score.toFixed(2)}`;
  const node = new SearchNode(label, vscode.TreeItemCollapsibleState.None);
  node.command = {
    command: "rag.openSearchTarget",
    title: "Open Location",
    arguments: [it.path, it.start_line, it.end_line]
  };
  node.tooltip = it.text_preview || "";
  return node;
}

class SearchNode extends vscode.TreeItem {
  constructor(label: string, collapsibleState: vscode.TreeItemCollapsibleState) {
    super(label, collapsibleState);
  }
}

export function registerSearchOpenCommand(ctx: vscode.ExtensionContext) {
  ctx.subscriptions.push(
    vscode.commands.registerCommand("rag.openSearchTarget", async (p: string, s?: number, e?: number) => {
      await openLocation(p, s, e);
    })
  );
}
