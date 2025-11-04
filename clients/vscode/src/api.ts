import axios, { AxiosInstance } from "axios";
import * as vscode from "vscode";

export type SearchItem = {
  id: string;
  score: number;
  name: string;
  kind: string;
  path: string;
  start_line: number;
  end_line: number;
  text_preview?: string;
};

export type SearchResponse = {
  query: string;
  total: number;
  results: SearchItem[];
};

export type Evidence = {
  id: string;
  path: string;
  name: string;
  kind: string;
  start_line: number;
  end_line: number;
  score: number;
};

export type ExplainResponse = {
  query: string;
  answer: string;
  evidences: Evidence[];
  timings_ms: { [k: string]: number };
  model: string;
  provider: string;
  usage?: Record<string, unknown>;
};

export type PingResponse = {
  ok: boolean;
  provider: string;
  model: string;
};

function getCfg() {
  const cfg = vscode.workspace.getConfiguration("rag");
  return {
    apiBase: cfg.get<string>("apiBase") ?? "http://127.0.0.1:8000",
    topK: cfg.get<number>("topK") ?? 8,
    symbolBoost: cfg.get<number>("symbolBoost") ?? 2.0,
    providerOverride: cfg.get<string>("providerOverride") ?? "auto",
    maxTokens: cfg.get<number>("maxTokens") ?? 600,
    maxCtxChars: cfg.get<number>("maxCtxChars") ?? 6000
  };
}

let client: AxiosInstance | null = null;
function http(): AxiosInstance {
  const { apiBase } = getCfg();
  if (!client) {
    client = axios.create({
      baseURL: apiBase,
      timeout: 60000  // 60s，兼容本地 LLM 慢一点
    });
  } else {
    (client.defaults.baseURL as any) = apiBase;
  }
  return client;
}

export async function ping(): Promise<PingResponse> {
  try {
    const res = await http().get<PingResponse>("/ping");
    return res.data;
  } catch {
    return { ok: false, provider: "offline", model: "offline" } as any;
  }
}

export async function search(query: string): Promise<SearchResponse> {
  const { topK, symbolBoost } = getCfg();
  const body = { query, top_k: topK, symbol_boost: symbolBoost };
  const res = await http().post<SearchResponse>("/search", body);
  return res.data;
}

export async function explain(query: string): Promise<ExplainResponse> {
  const { topK, maxTokens, maxCtxChars, providerOverride } = getCfg();
  const body: any = { query, top_k: topK, max_tokens: maxTokens, max_ctx_chars: maxCtxChars };
  if (providerOverride && providerOverride !== "auto") body.provider = providerOverride;
  const res = await http().post<ExplainResponse>("/explain", body);
  return res.data;
}

export function friendlyError(e: any): string {
  if (axios.isAxiosError(e)) {
    const code = e.response?.status;
    const data = e.response?.data;
    if (code && data) return `API ${code}: ${typeof data === "string" ? data : JSON.stringify(data)}`;
    if (e.code === "ECONNREFUSED") return "Cannot connect to RAG API (connection refused).";
    if (e.code === "ECONNABORTED") return "RAG API timeout.";
  }
  return (e?.message || String(e));
}
