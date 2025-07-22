import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def interactive_query():
    """交互式查询向量数据库"""
    
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
    
    print("✅ 数据库加载完成！")
    print("💡 您可以输入任何问题来搜索相关内容")
    print("💡 输入 'quit' 或 'exit' 退出程序")
    print("=" * 50)
    
    while True:
        # 获取用户输入
        query = input("\n🤔 请输入您的问题: ").strip()
        
        # 检查退出条件
        if query.lower() in ['quit', 'exit', '退出', 'q']:
            print("👋 再见！")
            break
        
        if not query:
            print("⚠️ 请输入有效的问题！")
            continue
        
        try:
            print(f"\n🔍 正在搜索: '{query}'")
            
            # 执行相似性搜索
            results = vectorstore.similarity_search_with_score(query, k=3)
            
            if results:
                print(f"✅ 找到 {len(results)} 个相关结果:\n")
                
                for i, (doc, score) in enumerate(results):
                    print(f"📄 结果 {i+1} (相似度: {1-score:.3f}):")
                    print(f"   📁 来源: {doc.metadata.get('source', '未知')}")
                    print(f"   📄 页码: {doc.metadata.get('page', '未知')}")
                    print(f"   📝 内容:")
                    
                    # 格式化显示内容
                    content_lines = doc.page_content.strip().split('\n')
                    for line in content_lines:
                        if line.strip():
                            print(f"      {line.strip()}")
                    
                    print("-" * 40)
            else:
                print("❌ 没有找到相关结果，请尝试其他关键词。")
                
        except Exception as e:
            print(f"❌ 搜索出错: {e}")
            print("请检查Ollama是否正在运行，以及nomic-embed-text模型是否可用。")

def quick_search(query, top_k=3):
    """快速搜索函数，可以在其他脚本中调用"""
    
    if not os.path.exists("./chroma_db"):
        return None
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    
    return vectorstore.similarity_search(query, k=top_k)

if __name__ == "__main__":
    interactive_query() 