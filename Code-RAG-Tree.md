Code-RAG/
├── .env                    # (需手动创建) 环境变量：API Key, 模型路径等
├── .dockerignore           # (新增) Docker 构建忽略规则
├── Dockerfile              # (新增) FastAPI 后端镜像构建文件
├── docker-compose.yml      # (新增) 一键拉起 API + Local LLM + Redis
├── requirements.txt        # Python 项目依赖列表
├── README.md               # 项目文档
│
├── api/                    # [服务层] FastAPI 后端入口
│   └── main.py             # 定义 /search, /explain 路由与 API 逻辑
│
├── ai/                     # [模型层] LLM 统一接口
│   └── llm.py              # 封装 OpenAI / Local (llama.cpp) 调用切换
│
├── indexer/                # [写入层] 索引构建与入库 (RAG 核心)
│   ├── ast_dump.py         # Tree-sitter 解析：代码 -> AST
│   ├── chunker.py          # 切片逻辑：AST -> 语义 Chunks
│   └── embed_ingest.py     # 向量化：Chunks -> Embedding -> ChromaDB
│
├── retriever/              # [读取层] 检索逻辑
│   └── hybrid_search.py    # 混合检索算法 (向量相似度 + 符号/路径加权)
│
├── clients/                # [应用层] 前端与插件
│   └── vscode/             # VS Code 插件工程
│       ├── package.json    # 插件配置与命令定义
│       └── src/            # TypeScript 源码
│           ├── extension.ts    # 插件入口
│           ├── api.ts          # 与后端 HTTP 通信
│           └── explainPanel.ts # 侧边栏 WebView 渲染
│
├── configs/                # [配置层]
│   ├── embedding.yaml      # 嵌入模型配置
│   └── retrieval.yaml      # 检索参数配置
│
├── data/                   # [持久化数据] (Docker 挂载卷)
│   └── chroma_db/          # Chroma 向量数据库文件
│
├── models/                 # [模型仓库] (Docker 挂载卷)
│   └── qwen2.5-*.gguf      # 本地 GGUF 模型文件
│
├── eval/                   # [评测体系]
│   ├── qa_min.jsonl        # 最小评测数据集 (QA对)
│   └── run_eval.py         # 自动化评测脚本 (Hit@K, MRR)
│
└── scripts/                # [运维脚本]
    ├── day7-up.sh          # (旧) Shell 启动脚本
    └── doctor.sh           # 环境自检脚本