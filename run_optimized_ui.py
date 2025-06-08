"""
Optimized UI for unified financial data system using Gradio
"""

import os
import warnings
import gradio as gr
from pathlib import Path
import numpy as np
import sys
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xlm.utils.unified_data_loader import UnifiedDataLoader
from xlm.registry.generator import load_generator
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.retriever.sbert_retriever import SBERTRetriever
from xlm.components.encoder.encoder import Encoder

# Ignore warnings
warnings.filterwarnings("ignore")

def ensure_directories():
    """Ensure required directories exist"""
    dirs = [
        "data",
        "data/processed",
        "data/tatqa_dataset_raw",
        "data/alphafin"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def is_chinese(text):
    """检测文本是否包含中文"""
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))

def load_model():
    model_name = "facebook/opt-1.3b"
    cache_dir = "D:\\AI\\huggingface"
    
    try:
        # First try loading from local cache
        print(f"Loading model from local cache: {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,  # Only use local files
            use_fast=True  # Use fast tokenizer
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            cache_dir=cache_dir,
            local_files_only=True,  # Only use local files
            low_cpu_mem_usage=True  # Optimize memory usage
        )
        print("Model loaded successfully from local cache")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please ensure the model is downloaded to the correct location:")
        print(f"1. Model should be in: {cache_dir}")
        print("2. You can download the model manually using:")
        print("   git lfs install")
        print(f"   git clone https://huggingface.co/{model_name} {cache_dir}/{model_name}")
        raise e
    
    return model, tokenizer

def generate_response(model, tokenizer, query, history=None):
    if history is None:
        history = []
    
    try:
        # Format the prompt
        prompt = query
        if history:
            prompt = "\n".join([f"Human: {h[0]}\nAssistant: {h[1]}" for h in history])
            prompt += f"\nHuman: {query}\nAssistant:"
        
        # Tokenize with fixed input_ids
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(model.device)
        
        # Generate with minimal parameters
        outputs = model.generate(
            input_ids,
            max_new_tokens=32,  # Further reduced for faster response
            do_sample=False,  # Disable sampling for deterministic output
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip(), history + [(query, response.strip())]
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        return f"Error: {str(e)}", history

class OptimizedUI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.data_loader = UnifiedDataLoader(
            batch_size=16,  # 减小批处理大小
            max_samples=500  # 减少每个数据源的样本数
        )
        # 数据加载器会在初始化时自动加载数据
        self.documents = self.data_loader.documents
        self.rag_system = None
        
    def init_model(self):
        """Initialize the model and system components"""
        if self.model is not None:
            return
            
        try:
            # 初始化检索器
            print("Initializing retriever...")
            encoder = Encoder(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir="D:/AI/huggingface"
            )
            retriever = SBERTRetriever(
                encoder=encoder,
                corpus_documents=self.documents
            )
            
            # 初始化生成器
            print("Initializing generator...")
            generator = load_generator(
                generator_model_name="facebook/opt-125m",
                use_local_llm=True
            )
            
            # 初始化RAG系统
            print("Initializing RAG system...")
            self.rag_system = RagSystem(
                retriever=retriever,
                generator=generator,
                prompt_template="Context: {context}\nQuestion: {question}\nAnswer:",
                retriever_top_k=1
            )
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            traceback.print_exc()
            raise e

    def process_query(self, query: str, datasource: str):
        """Process a query and return results"""
        try:
            # 运行RAG系统
            rag_output = self.rag_system.run(user_input=query)
            
            # 准备输出
            answer = rag_output.generated_responses[0]
            doc = rag_output.retrieved_documents[0]
            score = rag_output.retriever_scores[0]
            
            # 构建响应
            response = {
                "answer": answer,
                "context": f"相关度分数: {score:.4f}\n\n{doc.content}"
            }
            
            return response["answer"], response["context"]
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            traceback.print_exc()
            return "处理查询时发生错误", str(e)

    def run(self):
        """Run the Gradio interface"""
        try:
            self.init_model()
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return
        
        with gr.Blocks() as demo:
            gr.Markdown("# Financial Explainable RAG System")
            
            # 输入区域
            with gr.Row():
                with gr.Column(scale=4):
                    datasource = gr.Radio(
                        choices=["TatQA", "AlphaFin", "Both"],
                        value="Both",
                        label="Data Source"
                    )
                    
            with gr.Row():
                with gr.Column(scale=4):
                    query = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your question",
                        label="Question"
                    )
                    submit_btn = gr.Button("Submit")
            
            # 使用标签页分离显示
            with gr.Tabs():
                # 回答标签页
                with gr.TabItem("Answer"):
                    answer_box = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        label="Answer",
                        lines=5
                    )
                
                # 解释标签页
                with gr.TabItem("Explanation"):
                    context_box = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        label="Context",
                        lines=10
                    )

            # 添加示例问题
            gr.Examples(
                examples=[
                    ["What is the revenue for Q4 2019?"],
                    ["What is the operating margin in 2018?"],
                    ["What are the R&D expenses in 2019?"],
                    ["2019年第四季度利润是多少？"],
                    ["毛利率趋势分析"],
                    ["研发投入比例"]
                ],
                inputs=query,
                label="Example Questions"
            )

            # 处理提交
            submit_btn.click(
                fn=self.process_query,
                inputs=[query, datasource],
                outputs=[answer_box, context_box],
            )

        demo.launch(share=False)

if __name__ == "__main__":
    # 确保必要的目录存在
    ensure_directories()
    
    # 启动UI
    print("Starting Financial QA System...")
    ui = OptimizedUI()
    ui.run() 