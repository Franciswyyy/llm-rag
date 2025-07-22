# app.py
import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 设置页面配置
st.set_page_config(
    page_title="📚 RAG 知识问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    .bot-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    .source-info {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vectorstore():
    """加载向量数据库"""
    if not os.path.exists("./chroma_db"):
        st.error("❌ 向量数据库不存在！请先运行 ingest.py 构建知识库。")
        st.stop()
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

@st.cache_resource
def setup_qa_chain():
    """设置问答链"""
    vectorstore = load_vectorstore()
    
    # 创建LLM
    llm = OllamaLLM(model="llama3.1:latest", temperature=0.7)
    
    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # 创建提示模板
    prompt_template = """
你是一个专业的金融知识助手。请基于以下提供的上下文信息来回答用户的问题。

上下文信息：
{context}

用户问题：{question}

请注意：
1. 只基于提供的上下文信息来回答问题
2. 如果上下文中没有相关信息，请诚实地说明
3. 回答要准确、详细且易于理解
4. 可以结合上下文提供具体的例子或建议

回答：
"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # 创建QA链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def display_source_documents(source_docs, message_id):
    """显示来源文档"""
    if source_docs:
        st.markdown("### 📖 参考来源")
        for i, doc in enumerate(source_docs):
            with st.expander(f"来源 {i+1} - {doc.metadata.get('source', '未知来源')}"):
                st.markdown(f"**页码：** {doc.metadata.get('page', '未知')}")
                st.markdown(f"**内容：**")
                st.text_area(
                    "文档内容",
                    value=doc.page_content,
                    height=200,
                    key=f"source_{message_id}_{i}_{hash(doc.page_content[:50])}",
                    disabled=True
                )

def main():
    # 标题
    st.markdown("""
    <div class="main-header">
        <h1>📚 RAG 知识问答系统</h1>
        <p>基于《富爸爸穷爸爸》的智能问答助手</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.markdown("## ⚙️ 系统设置")
        
        # 检查系统状态
        st.markdown("### 📊 系统状态")
        
        # 检查向量数据库
        if os.path.exists("./chroma_db"):
            st.success("✅ 向量数据库已加载")
            try:
                vectorstore = load_vectorstore()
                collection = vectorstore._collection
                doc_count = collection.count()
                st.info(f"📄 文档片段数量: {doc_count}")
            except Exception as e:
                st.error(f"❌ 数据库连接失败: {e}")
        else:
            st.error("❌ 向量数据库未找到")
            st.markdown("请先运行以下命令构建知识库：")
            st.code("python ingest.py")
        
        # 模型设置
        st.markdown("### 🤖 模型配置")
        st.info("🔗 嵌入模型: nomic-embed-text")
        st.info("🧠 语言模型: llama3.1:latest")
        
        # 使用说明
        st.markdown("### 💡 使用说明")
        st.markdown("""
        1. 在右侧输入框中输入您的问题
        2. 点击发送或按Enter键
        3. 系统会搜索相关文档并生成回答
        4. 查看参考来源了解答案依据
        """)
        
        # 示例问题
        st.markdown("### 🎯 示例问题")
        example_questions = [
            "什么是财务自由？",
            "富人和穷人的思维有什么不同？",
            "如何开始投资？",
            "什么是资产和负债？",
            "如何建立被动收入？"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                st.session_state.example_question = question
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 💬 智能问答")
        
        # 初始化会话状态
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "qa_chain" not in st.session_state:
            try:
                with st.spinner("🔄 正在加载问答系统..."):
                    st.session_state.qa_chain = setup_qa_chain()
                st.success("✅ 问答系统加载完成！")
            except Exception as e:
                st.error(f"❌ 系统加载失败: {e}")
                st.stop()
        
        # 显示聊天历史
        for msg_idx, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🤔 您：</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 助手：</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # 显示来源文档
                if "sources" in message:
                    display_source_documents(message["sources"], msg_idx)
        
        # 输入框
        user_question = st.text_input(
            "请输入您的问题：",
            key="user_input",
            placeholder="例如：什么是财务自由？",
            value=st.session_state.get("current_input", "")
        )
        
        # 发送按钮
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            send_button = st.button("📤 发送", use_container_width=True)
        
        with col_clear:
            clear_button = st.button("🗑️ 清空对话", use_container_width=True)
        
        # 处理示例问题
        example_triggered = False
        if "example_question" in st.session_state:
            user_question = st.session_state.example_question
            del st.session_state.example_question
            example_triggered = True
        
        # 处理用户输入 - 只在明确的操作时触发
        if (send_button or example_triggered) and user_question and user_question.strip():
            # 添加用户消息
            st.session_state.messages.append({
                "role": "user", 
                "content": user_question
            })
            
            # 生成回答
            with st.spinner("🔍 正在搜索相关信息并生成回答..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": user_question})
                    answer = result["result"]
                    source_docs = result.get("source_documents", [])
                    
                    # 添加助手回答
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_docs
                    })
                    
                    # 清空输入框
                    st.session_state.current_input = ""
                    
                    # 页面会自动重新渲染显示新消息
                    
                except Exception as e:
                    st.error(f"❌ 生成回答时出错: {e}")
                    st.markdown("请检查：")
                    st.markdown("1. Ollama是否正在运行")
                    st.markdown("2. llama3.1:latest模型是否已下载")
                    st.markdown("3. nomic-embed-text模型是否可用")
        
        # 清空对话
        if clear_button:
            st.session_state.messages = []
            st.session_state.current_input = ""
    
    with col2:
        st.markdown("## 📈 统计信息")
        
        if os.path.exists("./chroma_db"):
            try:
                vectorstore = load_vectorstore()
                collection = vectorstore._collection
                
                # 基本统计
                st.metric("📄 文档片段", collection.count())
                st.metric("💬 对话轮数", len(st.session_state.messages) // 2)
                
                # 最近查询
                if st.session_state.messages:
                    st.markdown("### 🕐 最近查询")
                    user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
                    for msg in user_messages[-3:]:
                        st.markdown(f"• {msg['content'][:50]}...")
                
            except Exception as e:
                st.error(f"获取统计信息失败: {e}")

if __name__ == "__main__":
    main()