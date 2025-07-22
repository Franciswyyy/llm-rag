import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. 定义文档加载器，从'docs'文件夹加载PDF
loader = DirectoryLoader('./docs/', glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# 2. 创建文本切分器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 3. 创建嵌入模型实例 (使用Ollama)
# 它会自动使用您在Ollama中运行的nomic-embed-text模型
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. 创建并持久化存储向量数据库
# 这会处理所有嵌入并存入本地的'chroma_db'文件夹
Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")

print("知识库构建完成！") 