import gradio as gr
from xlm.components.encoder.encoder import Encoder
from xlm.registry.generator import load_generator
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.retriever.sbert_retriever import SBERTRetriever
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

# 初始化RAG系统
def init_rag_system():
    # 配置模型和端点
    encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    generator_model_name = "facebook/opt-1.3b"
    
    # 加载知识库文档
    with open("data/nfl_panthers.txt", encoding="utf-8") as f:
        content = f.read().strip()
    
    # 创建文档对象
    doc = DocumentWithMetadata(
        content=content,
        metadata=DocumentMetadata(
            source="nfl_panthers.txt",
            created_at="",
            author=""
        )
    )
    corpus_documents = [doc]

    # 设置提示模板
    prompt_template = """Based on the following context, please answer the question. Only provide the direct answer without any additional questions.
Context: {context}
Question: {question}
Answer: """

    # 初始化组件
    encoder = Encoder(model_name=encoder_model_name)
    retriever = SBERTRetriever(encoder=encoder, corpus_documents=corpus_documents)
    
    # 使用本地LLM
    generator = load_generator(
        generator_model_name=generator_model_name,
        use_local_llm=True
    )

    # 创建RAG系统
    return RagSystem(
        retriever=retriever,
        generator=generator,
        prompt_template=prompt_template,
        retriever_top_k=1,
    )

# 初始化系统
system = init_rag_system()

def process_question(question):
    # 运行系统并获取结果
    rag_output = system.run(user_input=question)
    
    # 构建响应
    retrieved_docs = []
    for doc, score in zip(rag_output.retrieved_documents, rag_output.retriever_scores):
        retrieved_docs.append(f"相关度分数: {score:.4f}\n内容: {doc.content}\n")
    
    retrieved_text = "\n".join(retrieved_docs)
    answer = rag_output.generated_responses[0] if rag_output.generated_responses else "无法生成答案"
    
    return f"检索到的文档:\n{retrieved_text}\n生成的答案:\n{answer}"

# 创建Gradio界面
iface = gr.Interface(
    fn=process_question,
    inputs=gr.Textbox(label="请输入你的问题", placeholder="例如：How many points did the Panthers defense surrender?"),
    outputs=gr.Textbox(label="回答"),
    title="Panthers RAG 问答系统",
    description="这是一个基于RAG的问答系统，可以回答关于Panthers的问题。系统会先检索相关文档，然后基于检索到的内容生成答案。",
    examples=[
        ["How many points did the Panthers defense surrender?"],
        ["Who led the team in sacks?"],
        ["How many interceptions did the Panthers have?"]
    ]
)

if __name__ == "__main__":
    print("正在启动Web界面...")
    iface.launch(share=False) 