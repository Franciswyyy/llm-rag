# RAG 知识库构建

这是一个基于LangChain和Ollama的RAG（检索增强生成）知识库构建项目。

## 安装依赖

```bash
pip install -r requirements.txt
```
123123
## 使用前准备1

1. 确保已安装并运行Ollama
2. 下载所需模型：
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.1:latest
   ```

## 使用流程

### 步骤1: 设置环境
```bash
# 激活虚拟环境
source rag_env/bin/activate
```

### 步骤2: 启动Ollama并下载模型
```bash
# 确保Ollama服务运行（在新终端窗口）
ollama serve

# 下载所需模型
ollama pull nomic-embed-text
ollama pull llama3.1:latest
```

> 📝 **注意**：如果您已经有其他Ollama模型（如llama3.1、llama3.2、qwen2.5等），可以修改`app.py`中的模型名称来使用它们。

### 步骤3: 构建知识索引
将您的PDF文档放在`docs/`文件夹中，然后运行：

```bash
python ingest.py
```

脚本将：
1. 加载`docs/`文件夹中的所有PDF文档
2. 将文档切分成1000字符的小块（重叠100字符）
3. 使用Ollama的nomic-embed-text模型生成嵌入向量
4. 将向量存储到本地的`chroma_db/`文件夹中

### 步骤4: 启动应用
```bash
# 启动Web应用（推荐）
streamlit run app.py

# 或者使用命令行查询
python query_db.py
```

## 查看数据库

构建完知识库后，您可以通过以下方式查看和测试：

### 查看数据库基本信息
```bash
python view_db.py
```

此脚本会显示：
- 数据库中的文档片段总数
- 文档来源信息
- 示例文档片段内容
- 简单的相似性搜索测试

### 交互式查询测试
```bash
python query_db.py
```

此脚本提供交互式界面，您可以：
- 输入任意问题进行搜索
- 查看相关文档片段和相似度分数
- 测试不同查询的效果

### 启动Web应用界面
```bash
streamlit run app.py
```

这将启动一个现代化的Web界面，提供：
- 🎨 美观的用户界面
- 💬 实时聊天对话
- 📖 来源文档展示
- 📊 系统状态监控
- 🎯 示例问题快速测试
- 📈 使用统计信息

## 文件结构

```
llm-rag/
├── docs/                    # 存放PDF文档
├── chroma_db/              # 向量数据库（运行后生成）
├── rag_env/                # Python虚拟环境
├── app.py                  # Streamlit Web应用
├── ingest.py               # 知识索引构建脚本
├── view_db.py              # 数据库查看脚本
├── query_db.py             # 交互式查询脚本
├── requirements.txt        # Python依赖
└── README.md              # 使用说明
``` 
