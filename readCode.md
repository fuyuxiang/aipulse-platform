# RAGFlow 代码详细分析文档

## 一、项目概述

RAGFlow 是一个基于深度文档理解的开源 RAG（检索增强生成）引擎。它是一个全栈应用，具有以下特点：

- **后端**：Python + Flask/Quart API 服务
- **前端**：React/TypeScript + UmiJS
- **微服务架构**：Docker 部署
- **多数据存储**：MySQL、Elasticsearch/Infinity、Redis、MinIO

---

## 二、整体架构

### 2.1 目录结构

| 目录 | 功能说明 |
|------|----------|
| `/api/` | Flask REST API 服务 |
| `/api/db/` | 数据库模型和服务层 |
| `/deepdoc/` | 文档解析（OCR、布局识别、PDF解析） |
| `/rag/` | RAG 核心引擎（检索、分块、LLM调用） |
| `/agent/` | Agent 工作流引擎 |
| `/web/` | React 前端（UmiJS） |
| `/graphrag/` | 知识图谱 RAG |
| `/memory/` | 记忆系统 |
| `/plugin/` | 插件系统 |
| `/common/` | 公共常量和工具 |
| `/conf/` | 配置文件 |

### 2.2 后端入口和启动流程

**主入口文件**：`api/ragflow_server.py`

启动流程：
1. 初始化日志系统 (`init_root_logger`)
2. 初始化数据库 (`init_web_db()`)
3. 初始化Web数据 (`init_web_data()`)
4. 加载运行时配置 (`RuntimeConfig.init_env()`)
5. 加载插件 (`GlobalPluginManager.load_plugins()`)
6. 注册信号处理（SIGINT/SIGTERM）
7. 启动后台线程 (`update_progress`) 用于更新文档处理进度
8. 启动 Flask/Quart HTTP 服务器

**核心技术栈**：
- Web 框架：Quart（异步 Flask）
- 数据库：MySQL/PostgreSQL + Peewee ORM
- 向量存储：Elasticsearch/Infinity
- 缓存：Redis
- 文件存储：MinIO/S3

### 2.3 API 路由结构

API 入口：`api/apps/__init__.py`

路由通过动态加载 `*_app.py` 文件自动注册：

| 文件 | 路由前缀 | 功能 |
|------|----------|------|
| `kb_app.py` | `/v1/kb` | 知识库管理 |
| `document_app.py` | `/v1/document` | 文档管理 |
| `dialog_app.py` | `/v1/dialog` | 对话应用 |
| `conversation_app.py` | `/v1/conversation` | 对话会话 |
| `file_app.py` | `/v1/file` | 文件管理 |
| `canvas_app.py` | `/v1/canvas` | Agent 画布 |
| `llm_app.py` | `/v1/llm` | LLM 模型管理 |
| `tenant_app.py` | `/v1/tenant` | 租户管理 |
| `user_app.py` | `/v1/user` | 用户管理 |

---

## 三、数据库模型分析

### 3.1 核心数据表

核心表定义在 `api/db/db_models.py`：

| 表名 | 说明 |
|------|------|
| `user` | 用户 |
| `tenant` | 租户 |
| `user_tenant` | 用户-租户关联 |
| `knowledgebase` | 知识库 |
| `document` | 文档 |
| `file` | 文件/文件夹 |
| `file2document` | 文件-文档关联 |
| `task` | 解析任务 |
| `dialog` | 对话应用 |
| `conversation` | 对话会话 |
| `api_token` | API 令牌 |
| `user_canvas` | Agent 画布 |
| `canvas_template` | 画布模板 |
| `llm` | LLM 模型定义 |
| `llm_factories` | LLM 厂商 |
| `tenant_llm` | 租户 LLM 配置 |
| `mcp_server` | MCP 服务器 |
| `connector` | 数据源连接器 |
| `memory` | 记忆系统 |

### 3.2 表之间的关系

```
User <-- UserTenant --> Tenant
Tenant --(has many)--> Knowledgebase --(has many)--> Document
Tenant --(has many)--> Dialog --(has many)--> Conversation
Tenant --(has many)--> File
Tenant --(has many)--> UserCanvas
Knowledgebase --(has many)--> Task
```

---

## 四、文档处理流程（核心）

### 4.1 整体流程

```
用户上传文件 -> API 创建 Task -> Redis 任务队列 -> task_executor（任务执行器）-> 解析器（Parser）-> 分块（Chunking）-> 向量化（Embedding）-> 存储（ES/Infinity）
```

### 4.2 关键文件说明

#### 4.2.1 任务执行器 - task_executor.py

**文件位置**：`rag/svr/task_executor.py`

**作用**：文档处理的核心调度中心，负责从 Redis 队列中消费任务并协调整个处理流程。

**解析器注册表**（第 75-92 行）：
```python
FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,
    ParserType.KG.value: naive,
    ParserType.TAG.value: tag
}
```

**核心函数**：

1. **`build_chunks()` 函数**（第 224-480 行）
   - 从 MinIO 获取文件二进制内容
   - 根据 `parser_id` 从 FACTORY 获取解析器
   - 调用解析器的 `chunk()` 方法进行文档解析
   - 处理关键词提取 (`auto_keywords`)
   - 处理问题生成 (`auto_questions`)
   - 处理元数据生成 (`enable_metadata`)
   - 处理标签 (`tag_kb_ids`)

2. **`embedding()` 函数**（第 524-575 行）
   - 提取每个 chunk 的标题和内容
   - 批量调用 embedding 模型进行向量化
   - 根据 `filename_embd_weight` 计算加权向量
   - 将向量添加到 chunk 对象的 `q_{vector_size}_vec` 字段

3. **`insert_es()` 函数**（第 796-861 行）
   - 处理母 chunk（mom）的存储
   - 批量将 chunks 插入到 Elasticsearch/Infinity
   - 更新任务进度和 chunk IDs

#### 4.2.2 通用解析器 - naive.py

**文件位置**：`rag/app/naive.py`

**作用**：实现最通用的文档解析逻辑，是 FACTORY 中 "naive" 对应的解析器。

**支持的解析方法**：
```python
PARSERS = {
    "deepdoc": by_deepdoc,      # DeepDOC 布局分析 + OCR
    "mineru": by_mineru,        # MinerU 解析
    "docling": by_docling,      # Docling 解析
    "tcadp": by_tcadp,         # 腾讯云 TCADP 解析
    "plaintext": by_plaintext,  # 纯文本解析
}
```

**`by_deepdoc()` 函数**（第 46-61 行）使用 DeepDOC 进行 PDF 解析：
1. OCR 文字识别
2. 布局分析 (`_layouts_rec`)
3. 表格识别 (`_table_transformer_job`)
4. 文本合并 (`_text_merge`)

#### 4.2.3 分块处理 - naive_merge()

**文件位置**：`rag/nlp/__init__.py` 中的 `naive_merge()` 函数

**核心分块逻辑**：
1. 使用正则表达式按分隔符分割文本
2. 根据 `chunk_token_num` 限制每个 chunk 的 token 数量（默认 512）
3. 支持重叠分块 (`overlapped_percent`)
4. 支持自定义分隔符

#### 4.2.4 其他解析器

| 文件 | 位置 | 功能 |
|------|------|------|
| `paper.py` | `rag/app/paper.py` | 论文解析 |
| `book.py` | `rag/app/book.py` | 书籍解析 |
| `laws.py` | `rag/app/laws.py` | 法律文档解析 |
| `resume.py` | `rag/app/resume.py` | 简历解析 |
| `table.py` | `rag/app/table.py` | 表格解析 |
| `qa.py` | `rag/app/qa.py` | 问答解析 |
| `picture.py` | `rag/app/picture.py` | 图片/视频解析 |
| `audio.py` | `rag/app/audio.py` | 音频解析 |

### 4.3 深度文档解析 - deepdoc/

**文件位置**：`deepdoc/parser/`

| 文件 | 功能 |
|------|------|
| `pdf_parser.py` | PDF 解析（OCR、布局分析、表格识别） |
| `docx_parser.py` | Word 文档解析 |
| `excel_parser.py` | Excel 解析 |
| `markdown_parser.py` | Markdown 解析 |
| `html_parser.py` | HTML 解析 |
| `ppt_parser.py` | PPT 解析 |

**视觉处理**（`deepdoc/vision/`）：
- `layout_recognizer.py` - 布局识别
- `ocr.py` - OCR 文字识别
- `table_structure_recognizer.py` - 表格结构识别

### 4.4 数据流转过程

```
1. 用户上传文件 -> API 创建 Task 记录 -> TaskService.create_task()
2. Task 信息推送到 Redis 队列
3. task_executor 消费任务:
   a. TaskService.get_task() 获取任务详情
   b. 根据 parser_id 从 FACTORY 获取解析器
   c. File2DocumentService.get_storage_address() 获取 MinIO 地址
   d. get_storage_binary() 从 MinIO 下载文件
   e. 调用 chunker.chunk() 进行解析
   f. 调用 embedding() 进行向量化
   g. 调用 insert_es() 存储到 ES/Infinity
```

### 4.5 支持的解析方法

| 文件类型 | 支持的解析方法 |
|---------|--------------|
| PDF | deepdoc, mineru, docling, tcadp, plain_text, vlm |
| Excel | deepdoc, tcadp |
| Word | deepdoc (docx), tika (doc) |
| PPT | deepdoc, tcadp |
| 图片 | OCR, VLM |
| 音频 | SPEECH2TEXT |
| 文本 | 内置解析器 |

---

## 五、RAG 检索流程

### 5.1 检索核心文件

**文件位置**：`rag/nlp/search.py`

**核心类**：`Dealer`

### 5.2 检索流程

#### 5.2.1 检索入口

检索通过 `settings.retriever` 调用，实际执行函数为 `Dealer.retrieval()`（search.py:363-500）

**关键参数**：
```python
- question: 用户问题
- embd_mdl: Embedding模型
- tenant_ids: 租户ID列表
- kb_ids: 知识库ID列表
- similarity_threshold: 相似度阈值（默认0.2）
- vector_similarity_weight: 向量相似度权重（默认0.3）
- top_k: 返回topk结果（默认1024）
- rerank_mdl: 重排序模型（可选）
```

#### 5.2.2 检索步骤

1. **构建请求**（search.py:386-396）
   - 将问题转换为请求字典
   - 设置分页、topk、相似度阈值等参数

2. **向量搜索 + 关键词搜索融合**（search.py:401-134）
   - 使用 Embedding 模型将问题转换为向量
   - 调用 `self.search()` 执行混合搜索
   - 关键词匹配 + 向量匹配 + 权重融合（默认 0.05, 0.95）

3. **重排序**（search.py:404-428）
   - 如果有 rerank_mdl：使用模型重排序
   - 否则：使用内置混合相似度计算（token相似度 + 向量相似度）

4. **结果处理**（search.py:430-495）
   - 按相似度排序
   - 过滤低于阈值的 chunk
   - 返回分页结果
   - 聚合文档统计

#### 5.2.3 返回数据结构

```python
{
    "total": int,              # 匹配的chunk总数
    "chunks": [                # chunk列表
        {
            "chunk_id": str,
            "content_ltks": str,
            "content_with_weight": str,
            "doc_id": str,
            "docnm_kwd": str,
            "similarity": float,
            "vector_similarity": float,
            "term_similarity": float,
        }
    ],
    "doc_aggs": [             # 文档聚合
        {"doc_name": str, "doc_id": str, "count": int}
    ]
}
```

### 5.3 检索增强功能

RAGFlow 支持多种检索增强：

| 功能 | 说明 |
|------|------|
| TOC 增强 | 通过目录结构增强检索 |
| 子 chunk 获取 | 获取父 chunk 下的子 chunk |
| 知识图谱 | 使用知识图谱增强检索 |
| Tavily 搜索 | 外部网络搜索 |

---

## 六、对话/问答处理

### 6.1 核心文件

- **API**：`api/apps/dialog_app.py`
- **服务**：`api/db/services/dialog_service.py`
- **提示词**：`rag/prompts/generator.py`

### 6.2 问答流程（dialog_service.py:async_chat）

#### 6.2.1 请求处理

1. **检查配置**（dialog_service.py:283-286）
   - 如果没有知识库且没有 tavily_api_key，直接使用 LLM 回答

2. **模型初始化**（dialog_service.py:310-314）
   - 获取 chat_llm, embedding 模型, rerank 模型, tts 模型

3. **问题处理**（dialog_service.py:343-362）
   - 多轮对话优化问题
   - 跨语言处理
   - 关键词提取
   - 元数据过滤

#### 6.2.2 检索阶段（dialog_service.py:370-428）

```python
kbinfos = retriever.retrieval(
    question,
    embd_mdl,
    tenant_ids,
    kb_ids,
    1,
    top_n,
    similarity_threshold,
    vector_similarity_weight,
    doc_ids=attachments,
    top=top_k,
    aggs=True,
    rerank_mdl=rerank_mdl,
    rank_feature=label_question(question, kbs),
)
```

#### 6.2.3 提示词构建（dialog_service.py:444-449）

```python
# 知识库内容填充
knowledges = kb_prompt(kbinfos, max_tokens)

# 系统提示词
msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]

# 添加引用提示
prompt4citation = citation_prompt()

# 消息历史
msg.extend(messages[1:])
```

#### 6.2.4 LLM 生成（dialog_service.py:456-530）

```python
# 流式输出
async for ans in chat_mdl.async_chat_streamly(prompt, messages):
    # 引用插入
    answer = retriever.insert_citations(answer, chunks, vectors, embd_mdl)
    yield ans
```

---

## 七、Agent Canvas 系统

### 7.1 核心文件

| 文件 | 功能 |
|------|------|
| `agent/canvas.py` | Canvas 核心（Graph 类） |
| `api/apps/canvas_app.py` | Canvas API |
| `agent/component/base.py` | 组件基类 |
| `agent/tools/retrieval.py` | 检索组件 |
| `agent/component/llm.py` | LLM 组件 |
| `agent/component/categorize.py` | 分类组件 |

### 7.2 DSL 结构（canvas.py:42-78）

```python
dsl = {
    "components": {
        "component_id": {
            "obj": {
                "component_name": "组件类型",
                "params": {...}
            },
            "downstream": ["下游组件ID"],
            "upstream": ["上游组件ID"],
            "parent_id": "父组件ID(循环/分支)"
        }
    },
    "history": [],           # 对话历史
    "path": ["begin"],       # 执行路径
    "retrieval": {...},      # 检索结果
    "globals": {             # 全局变量
        "sys.query": "",
        "sys.user_id": "",
    }
}
```

### 7.3 执行流程（canvas.py:run()）

1. **初始化**（canvas.py:81-89）
   - 加载 DSL 配置
   - 初始化组件对象

2. **执行循环**（canvas.py: ~300-640）
   - 遍历执行路径
   - 获取上游输出作为输入
   - 调用组件的 `invoke_async()`
   - 检查错误/分支
   - 根据组件类型决定下一步

3. **流式输出**（canvas.py: ~400-550）
   - 支持实时流式输出
   - 处理引用、附件等

### 7.4 组件类型

| 组件 | 说明 |
|------|------|
| `Begin` | 流程起点 |
| `Retrieval` | 知识检索 |
| `LLM/Generate` | LLM 生成 |
| `Categorize/Switch` | 条件分支 |
| `Message` | 消息输出 |
| `Iteration/Loop` | 循环 |
| `Fillup` | 用户输入 |
| `Invoke` | 调用其他 Agent |

### 7.5 组件基类（component/base.py）

```python
class ComponentBase(ABC):
    component_name: str

    def __init__(self, canvas, id, param):
        self._canvas = canvas  # 指向Graph对象
        self._id = id
        self._param = param

    def invoke_async(self, **kwargs) -> dict:
        # 异步调用入口

    def output(self, var_nm=None) -> Union[dict, Any]:
        # 获取输出变量

    def set_output(self, key, value):
        # 设置输出变量

    def get_input(self, key=None):
        # 获取输入变量
```

---

## 八、LLM 封装

### 8.1 核心文件

| 文件 | 功能 |
|------|------|
| `rag/llm/chat_model.py` | 聊天模型 |
| `rag/llm/embedding_model.py` | Embedding 模型 |
| `rag/llm/rerank_model.py` | 重排序模型 |
| `rag/llm/cv_model.py` | 视觉模型（多模态） |
| `rag/llm/tts_model.py` | TTS 模型 |
| `rag/llm/stt_model.py` | 语音转文字模型 |

### 8.2 使用方式（LLMBundle）

**文件位置**：`api/db/services/llm_service.py`

```python
from api.db.services.llm_service import LLMBundle

# 获取聊天模型
chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)

# 同步调用
ans, tokens = chat_mdl.chat(system_prompt, history, gen_conf)

# 流式调用
async for chunk in chat_mdl.async_chat_streamly(system_prompt, history):
    print(chunk)

# 工具调用
chat_mdl.bind_tools(toolcall_session, tools)
ans, tokens = await chat_mdl.async_chat_with_tools(system, history, tools)
```

### 8.3 支持的 LLM 类型

通过工厂模式支持多种 LLM：

| 类名 | 厂商 |
|------|------|
| Base | 通用（OpenAI 兼容） |
| LiteLLMBase | LiteLLM 聚合 |
| XinferenceChat | Xinference |
| HuggingFaceChat | HuggingFace |
| BaiChuanChat | 百川 |
| GoogleChat | Google (Gemini/Claude) |
| MistralChat | Mistral |
| HunyuanChat | 腾讯混元 |
| SparkChat | 讯飞星火 |
| ZhipuChat | 智谱 |
| OllamaChat | Ollama |
| AnthropicChat | Anthropic (Claude) |
| GptV4 | OpenAI (GPT-4) |

### 8.4 支持的 Embedding 模型

| 类名 | 厂商 |
|------|------|
| OpenAIEmbed | OpenAI |
| AzureEmbed | Azure |
| OllamaEmbed | Ollama |
| XinferenceEmbed | Xinference |
| ZhipuEmbed | 智谱 |
| VoyageEmbed | Voyage |
| CohereEmbed | Cohere |
| NvidiaEmbed | NVIDIA |

---

## 九、服务层分析

### 9.1 服务目录

**位置**：`api/db/services/`

| 服务 | 职责 |
|------|------|
| `user_service.py` | 用户管理、认证、租户服务 |
| `knowledgebase_service.py` | 知识库 CRUD |
| `document_service.py` | 文档 CRUD、解析任务管理 |
| `file_service.py` | 文件上传/下载管理 |
| `task_service.py` | 解析任务队列 |
| `dialog_service.py` | 对话应用管理、LLM 调用 |
| `conversation_service.py` | 会话管理 |
| `llm_service.py` | LLM 模型封装 |
| `tenant_llm_service.py` | 租户 LLM 配置 |
| `canvas_service.py` | Agent 画布管理 |
| `search_service.py` | 搜索服务 |
| `connector_service.py` | 数据源连接器 |

### 9.2 服务调用关系

```
请求入口 (API)
    |
    v
[App Layer: *_app.py]
    |
    +---> [Service Layer: services/*.py]
    |           |
    |           +---> [DB Models: db_models.py]
    |           |
    |           +---> [External: LLM/Embedding/Rerank]
    |
    v
[Document Processing Pipeline]
    |
    +---> [deepdoc/] 文档解析
    |
    +---> [rag/flow/] 分块/向量化
    |
    v
[Storage]
    |
    +---> [MySQL] 原始数据
    +---> [Elasticsearch/Infinity] 向量索引
    +---> [MinIO/S3] 文件存储
    +---> [Redis] 缓存
```

---

## 十、图片/视频处理

### 10.1 图片处理

**文件位置**：`rag/app/picture.py`

**处理流程**：
1. **OCR 文字识别**（第 70-76 行）
   ```python
   bxs = ocr(np.array(img))
   txt = "\n".join([t[0] for _, t in bxs if t[0]])
   ```

2. **VLM 图片描述**（第 78-88 行）
   - 条件：只有当 OCR 文字 < 32 字符/单词时才调用 VLM
   - 调用 `cv_mdl.describe()` 获取图片描述

**支持的 VLM 模型**：
- GPT-4V (OpenAI)
- Claude Vision (Anthropic)
- Gemini Vision (Google)
- Moonshot Vision (Moonshot)

### 10.2 视频处理

**文件位置**：`rag/app/picture.py`（第 46-61 行）

**处理流程**：
```python
if any(filename.lower().endswith(ext) for ext in VIDEO_EXTS):
    cv_mdl = LLMBundle(tenant_id, llm_type=LLMType.IMAGE2TEXT, lang=lang)
    ans = asyncio.run(cv_mdl.async_chat(..., video_bytes=binary, filename=filename))
```

**支持的视频格式**：
`.mp4`, `.mov`, `.avi`, `.flv`, `.mpeg`, `.mpg`, `.webm`, `.wmv`, `.3gp`, `.3gpp`, `.mkv`

---

## 十一、二次开发指南

### 11.1 新增解析器

1. 在 `rag/app/` 下创建新的解析器文件（如 `video.py`）
2. 实现 `chunk()` 函数
3. 在 `rag/svr/task_executor.py` 的 FACTORY 字典中注册
4. 在 `common/constants.py` 的 ParserType 枚举中添加新类型

### 11.2 新增检索模式

1. 在 `rag/app/` 下创建新的检索模式文件
2. 实现检索逻辑
3. 在 `rag/nlp/search.py` 中集成

### 11.3 新增 Agent 组件

1. 在 `agent/component/` 下创建组件文件
2. 继承 `ComponentBase` 类
3. 实现 `invoke_async()` 方法
4. 在 `agent/component/__init__.py` 中注册

---

## 十二、总结

RAGFlow 是一个功能完整的 RAG 工程系统，具有以下特点：

1. **模块化设计**：清晰的 API 层、服务层、数据层分离
2. **多解析器支持**：支持 PDF、Word、Excel、PPT 等多种文档格式
3. **多种检索模式**：针对不同文档类型（论文、简历、法律等）优化
4. **灵活的 Agent 系统**：基于组件的工作流引擎
5. **多存储后端**：支持 Elasticsearch 和 Infinity 作为向量存储
6. **完整的租户体系**：支持多租户和团队协作
7. **多模态支持**：支持图片、视频、音频处理

---

*本文档由 Claude Code 自动生成*
*生成时间：2026-02-14*
*RAGFlow 版本：v0.23.0*
