from typing import List, Optional
import gradio as gr
from gradio.components import Markdown
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.encoder.encoder import Encoder
from xlm.registry.generator import load_generator
from xlm.components.retriever.sbert_retriever import SBERTRetriever
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata


class RagUI:
    def __init__(
        self,
        logo_path: str = "",
        css_path: str = "xlm/ui/css/demo.css",
        window_title: str = "Financial Explainable RAG System",
        title: str = "Financial Explainable RAG System",
        examples: Optional[List[List[str]]] = None,
    ):
        self.__logo_path = logo_path
        self.__css_path = css_path
        self.__window_title = window_title
        self.__title = title
        self.__examples = examples or [
            ["How many points did the Panthers defense surrender?"],
            ["Who led the team in sacks?"],
            ["How many interceptions did the Panthers have?"]
        ]
        
        # 初始化RAG系统
        self.__system = self.__init_rag_system()
        
        # 构建UI
        self.app: gr.Blocks = self.build_app()
        
    def __init_rag_system(self):
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

    def build_app(self):
        with gr.Blocks(
            theme=gr.themes.Monochrome().set(
                button_primary_background_fill="#009374",
                button_primary_background_fill_hover="#009374C4",
                checkbox_label_background_fill_selected="#028A6EFF",
            ),
            css=self.__css_path,
            title=self.__window_title,
        ) as demo:
            self.__build_app_title()
            user_input, system_response, submit_btn = self.__build_chat_interface()

            submit_btn.click(
                fn=self.run,
                inputs=[user_input],
                outputs=[system_response],
            )
            
            # 添加示例
            gr.Examples(
                examples=self.__examples,
                inputs=[user_input],
            )

        return demo

    def run(self, user_input: str):
        # 运行系统并获取结果
        rag_output = self.__system.run(user_input=user_input)
        
        # 构建响应
        retrieved_docs = []
        for doc, score in zip(rag_output.retrieved_documents, rag_output.retriever_scores):
            retrieved_docs.append(f"相关度分数: {score:.4f}\n内容: {doc.content}\n")
        
        retrieved_text = "\n".join(retrieved_docs)
        answer = rag_output.generated_responses[0] if rag_output.generated_responses else "无法生成答案"
        
        return f"检索到的文档:\n{retrieved_text}\n生成的答案:\n{answer}"

    def __build_app_title(self):
        with gr.Row():
            with gr.Column(scale=1):
                Markdown(
                    f'<p style="text-align: center; font-size:200%; font-weight: bold"'
                    f">{self.__title}"
                    f"</p>"
                )

    def __build_chat_interface(self):
        with gr.Row():
            with gr.Column(scale=1):
                user_input = gr.Textbox(
                    placeholder="Type your question here and press Enter.",
                    label="Question",
                    container=True,
                    lines=3,
                )

        with gr.Row():
            submit_btn = gr.Button(
                value="🔍 Ask",
                variant="secondary",
                elem_id="button",
                interactive=True,
            )

        with gr.Row():
            system_response = gr.Textbox(
                label="Answer",
                container=True,
                interactive=False,
                lines=10,
            )

        return user_input, system_response, submit_btn

    def launch(self, **kwargs):
        self.app.launch(**kwargs) 