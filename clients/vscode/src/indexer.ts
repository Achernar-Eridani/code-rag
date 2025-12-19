import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import * as crypto from "crypto";
import JSZip from "jszip";
import axios from "axios";
import FormData from "form-data";

const IGNORE_DIRS = new Set([
  "node_modules",
  ".git",
  "dist",
  "build",
  "out",
  "coverage",
  ".vscode",
  "__pycache__",
  ".venv",
  "venv",
  ".mypy_cache",
  ".pytest_cache",
  ".idea",
]);

const ALLOWED_EXTS = new Set([
  ".ts",
  ".js",
  ".tsx",
  ".jsx",
  ".py",
  ".java",
  ".go",
  ".rs",
  ".c",
  ".cpp",
  ".h",
  ".md",
  ".json",
  ".yaml",
  ".yml",
  ".toml",
]);

const MAX_FILE_SIZE = 1024 * 1024; // 1MB
const MAX_ZIP_SIZE = 20 * 1024 * 1024; // 20MB (soft limit)

/** Stable-ish workspace id: hash of the first workspace folder path. */
export function getWorkspaceId(): string {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) return "default";
  const fsPath = folders[0].uri.fsPath;
  const hash = crypto.createHash("sha1").update(fsPath).digest("hex");
  return hash.slice(0, 16);
}

function addFilesToZip(zip: JSZip, rootPath: string, currentPath: string) {
  const entries = fs.readdirSync(currentPath, { withFileTypes: true });

  for (const ent of entries) {
    const fullPath = path.join(currentPath, ent.name);

    if (ent.isDirectory()) {
      if (!IGNORE_DIRS.has(ent.name)) {
        addFilesToZip(zip, rootPath, fullPath);
      }
      continue;
    }

    if (!ent.isFile()) continue;

    const ext = path.extname(ent.name).toLowerCase();
    if (!ALLOWED_EXTS.has(ext)) continue;

    const stat = fs.statSync(fullPath);
    if (stat.size <= 0 || stat.size > MAX_FILE_SIZE) continue;

    const relPath = path.relative(rootPath, fullPath).replace(/\\/g, "/");
    const content = fs.readFileSync(fullPath);
    zip.file(relPath, content);
  }
}

async function pollJobStatus(apiBase: string, jobId: string, onTick?: (status: string) => void) {
  for (let i = 0; i < 300; i++) { // 10 min
    const res = await axios.get(`${apiBase}/index/status/${jobId}`);
    const status: string = res.data?.status ?? "unknown";
    onTick?.(status);
    if (status === "finished") return;
    if (status === "failed") throw new Error(res.data?.error || "Indexing job failed");
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error("Indexing timeout");
}

export async function buildAndUploadIndex(opts?: { fresh?: boolean }) {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    vscode.window.showErrorMessage("No workspace opened.");
    return;
  }

  const cfg = vscode.workspace.getConfiguration("rag");
  const apiBase =
    cfg.get<string>("apiBase") ||
    cfg.get<string>("api_base") ||
    cfg.get<string>("baseUrl") ||
    "http://localhost:8000";

  const rootPath = folders[0].uri.fsPath;
  const wsId = getWorkspaceId();
  const fresh = opts?.fresh ?? true;

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "RAG: Building workspace index...",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Zipping source code..." });

      const zip = new JSZip();
      addFilesToZip(zip, rootPath, rootPath);

      const zipBuf: Buffer = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });

      if (zipBuf.length > MAX_ZIP_SIZE) {
        throw new Error(
          `Workspace zip is too large (${Math.round(zipBuf.length / 1024 / 1024)}MB). ` +
            `Consider adding ignores or narrowing ALLOWED_EXTS.`
        );
      }

      progress.report({ message: "Uploading zip..." });

      const form = new FormData();
      form.append("file", zipBuf, { filename: "upload.zip", contentType: "application/zip" });
      form.append("workspace_id", wsId);
      form.append("fresh", fresh ? "true" : "false");

      const uploadRes = await axios.post(`${apiBase}/index/upload_and_build`, form, {
        headers: form.getHeaders(),
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
      });

      const jobId = uploadRes.data?.job_id;
      if (!jobId) throw new Error("Server did not return job_id.");

      progress.report({ message: "Indexing on server..." });

      await pollJobStatus(apiBase, jobId, (st) => {
        progress.report({ message: `Indexing on server... (${st})` });
      });
    }
  );
}
