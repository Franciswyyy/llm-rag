import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def view_database():
    """查看向量数据库的内容"""
    
    # 检查数据库是否存在
    if not os.path.exists("./chroma_db"):
        print("❌ 向量数据库不存在！请先运行 ingest.py 构建知识库。")
        return
    
    print("🔍 正在加载向量数据库...")
    
    # 创建嵌入模型实例
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 加载现有的向量数据库
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # 获取集合信息
    collection = vectorstore._collection
    
    print("\n📊 数据库基本信息:")
    print(f"   📄 文档片段总数: {collection.count()}")
    
    # 获取所有文档
    all_docs = vectorstore.get()
    
    if all_docs['documents']:
        print(f"   📝 文档总数: {len(set(doc.get('source', 'unknown') for doc in all_docs['metadatas']))}")
        
        # 显示前3个文档片段作为示例
        print("\n📋 文档片段示例:")
        for i, (doc, metadata) in enumerate(zip(all_docs['documents'][:3], all_docs['metadatas'][:3])):
            print(f"\n   片段 {i+1}:")
            print(f"   📁 来源: {metadata.get('source', '未知')}")
            print(f"   📄 页码: {metadata.get('page', '未知')}")
            print(f"   📝 内容预览: {doc[:200]}...")
            if len(doc) > 200:
                print("   ...")
        
        # 测试相似性搜索
        print("\n🔍 测试相似性搜索:")
        test_query = "财务自由"
        print(f"   查询: '{test_query}'")
        
        try:
            similar_docs = vectorstore.similarity_search(test_query, k=2)
            print(f"   找到 {len(similar_docs)} 个相关片段:")
            
            for i, doc in enumerate(similar_docs):
                print(f"\n   相关片段 {i+1}:")
                print(f"   📁 来源: {doc.metadata.get('source', '未知')}")
                print(f"   📄 页码: {doc.metadata.get('page', '未知')}")
                print(f"   📝 内容: {doc.page_content[:300]}...")
                
        except Exception as e:
            print(f"   ⚠️ 搜索测试失败: {e}")
    
    else:
        print("   ⚠️ 数据库中没有找到文档！")

if __name__ == "__main__":
    view_database() 