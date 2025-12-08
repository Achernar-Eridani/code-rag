import axios, { AxiosInstance } from "axios";
import * as vscode from "vscode";

// =============================================================================
//  Types (原有的定义)
// =============================================================================
// Agent tools
export type AgentToolResult = {
  path?: string;
  symbol?: string;
  kind?: string;
  start_line?: number;
  end_line?: number;
  score?: number;
  code?: string;
};

export type AgentExplainResponse = {
  query: string;
  answer: string;
  used_tool?: string | null;
  tool_input?: Record<string, any> | null;
  tool_results?: AgentToolResult[] | null;
};


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

// =============================================================================
//  Configuration & Client Logic
// =============================================================================

function getCfg() {
  const cfg = vscode.workspace.getConfiguration("rag");
  return {
    apiBase: cfg.get<string>("apiBase") ?? "http://127.0.0.1:8000",
    topK: cfg.get<number>("topK") ?? 8,
    symbolBoost: cfg.get<number>("symbolBoost") ?? 2.0,
    providerOverride: cfg.get<string>("providerOverride") ?? "auto",
    maxTokens: cfg.get<number>("maxTokens") ?? 600,
    maxCtxChars: cfg.get<number>("maxCtxChars") ?? 6000,
    // 新增：读取 API Key
    apiKey: cfg.get<string>("apiKey") ?? "",
  };
}

/**
 * 构造请求头：将配置转换为后端识别的 Header
 */
function buildHeaders(): Record<string, string> {
  const { providerOverride, apiKey } = getCfg();
  const headers: Record<string, string> = {};

  // 如果配置了 providerOverride 且不是 auto，通过 Header 告诉后端
  if (providerOverride && providerOverride !== "auto") {
    headers["x-llm-provider"] = providerOverride;
  }

  // 如果配置了 API Key，通过 Header 透传
  if (apiKey && apiKey.trim() !== "") {
    headers["x-api-key"] = apiKey.trim();
  }

  return headers;
}

let client: AxiosInstance | null = null;

function http(): AxiosInstance {
  const { apiBase } = getCfg();
  
  // 单例模式，但允许动态更新 apiBase
  if (!client) {
    client = axios.create({
      baseURL: apiBase.replace(/\/+$/, ""), // 自动去掉结尾的 /
      timeout: 60000, // 60s，兼容本地 LLM
    });
  } else {
    client.defaults.baseURL = apiBase.replace(/\/+$/, "");
  }
  
  return client;
}

// =============================================================================
//  API Functions
// =============================================================================

export async function ping(): Promise<PingResponse> {
  try {
    const headers = buildHeaders();
    // 这里的 headers 会携带 provider/model 信息，让 /ping 返回真实状态
    const res = await http().get<PingResponse>("/ping", { headers });
    return res.data;
  } catch {
    // 保持原有逻辑：连不上就返回 offline
    return { ok: false, provider: "offline", model: "offline" } as any;
  }
}

export async function search(query: string): Promise<SearchResponse> {
  const { topK, symbolBoost } = getCfg();
  const headers = buildHeaders();
  
  const body = { 
    query, 
    top_k: topK, 
    symbol_boost: symbolBoost 
  };
  
  const res = await http().post<SearchResponse>("/search", body, { headers });
  return res.data;
}

export async function explain(query: string): Promise<ExplainResponse> {
  const { topK, maxTokens, maxCtxChars } = getCfg();
  const headers = buildHeaders();

  const body = { 
    query, 
    top_k: topK, 
    max_tokens: maxTokens, 
    max_ctx_chars: maxCtxChars 
    // 注意：不再需要在 body 里传 provider，改走 headers
  };

  const res = await http().post<ExplainResponse>("/explain", body, { headers });
  return res.data;
}

export async function agentExplain(query: string): Promise<AgentExplainResponse> {
  const { maxTokens } = getCfg();
  const headers = buildHeaders();

  const body = {
    query,
    max_tokens: maxTokens,
  };

  const res = await http().post<AgentExplainResponse>("/agent/explain", body, { headers });
  return res.data;
}


export function friendlyError(e: any): string {
  if (axios.isAxiosError(e)) {
    const code = e.response?.status;
    const data = e.response?.data;
    if (code && data) {
      // 尝试解析后端返回的 detail 信息
      const msg = typeof data === "string" ? data : (data as any).detail || JSON.stringify(data);
      return `API ${code}: ${msg}`;
    }
    if (e.code === "ECONNREFUSED") return "Cannot connect to RAG API (connection refused). Is Docker running?";
    if (e.code === "ECONNABORTED") return "RAG API timeout (backend took too long).";
  }
  return (e?.message || String(e));
}