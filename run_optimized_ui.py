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
    def __init__(
        self,
        port=7860,
        share=False,
        debug=False,
        enable_queue=True,
    ):
        self.port = port
        self.share = share
        self.debug = debug
        self.enable_queue = enable_queue
        self.model = None
        self.tokenizer = None
        self.history = []
        
    def init_model(self):
        """Initialize the model and tokenizer"""
        if self.model is None:
            self.model, self.tokenizer = load_model()
            
    def chat(self, query, history=None):
        """Chat with the model"""
        if history is None:
            history = []
        response, new_history = generate_response(self.model, self.tokenizer, query, history)
        return response, new_history

    def run(self):
        """Run the Gradio interface"""
        try:
            self.init_model()
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return
        
        with gr.Blocks() as demo:
            gr.Markdown("# Financial Explainable RAG System")
            
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
            
            with gr.Row():
                gr.Markdown("## System Response")
                with gr.Column(scale=4):
                    answer_box = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        label="Answer",
                        max_lines=5
                    )
                    context_box = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        label="Context",
                        max_lines=5
                    )
            
            with gr.Row():
                gr.Markdown("## Example Questions")
                with gr.Column():
                    example_questions = gr.Examples(
                        examples=[
                            ["What is the revenue for Q4 2019?"],
                            ["What is the operating margin in 2018?"],
                            ["What are the R&D expenses in 2019?"],
                            ["2019年第四季度利润是多少？"],
                            ["毛利率趋势分析"],
                            ["研发投入比例"]
                        ],
                        inputs=query
                    )
            
            def process_query(question, source):
                try:
                    response, _ = self.chat(question, [])
                    return response, f"Source: {source}\nQuestion: {question}"
                except Exception as e:
                    print(f"Error in process_query: {str(e)}")
                    return f"Error: {str(e)}", "An error occurred while processing your request."
            
            submit_btn.click(
                process_query,
                inputs=[query, datasource],
                outputs=[answer_box, context_box],
                queue=self.enable_queue
            )
            
            query.submit(
                process_query,
                inputs=[query, datasource],
                outputs=[answer_box, context_box],
                queue=self.enable_queue
            )

        demo.queue()
        demo.launch(
            server_name="127.0.0.1",
            server_port=self.port,
            share=self.share,
            debug=self.debug
        )

if __name__ == "__main__":
    import sys
    import os
    import time
    import psutil
    import traceback
    
    def kill_python_processes():
        """Kill all Python processes except the current one"""
        current_pid = os.getpid()
        killed = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] == 'python.exe' and proc.pid != current_pid:
                    proc.kill()
                    killed.append(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return killed
    
    def is_port_in_use(port):
        """Check if a port is in use"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return False
            except socket.error:
                return True
    
    def wait_for_port_release(port, timeout=30):
        """Wait for a port to be released"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not is_port_in_use(port):
                return True
            time.sleep(1)
        return False
    
    try:
        # 显示系统信息
        print("\n=== System Information ===")
        print(f"Python version: {sys.version}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Process ID: {os.getpid()}")
        
        # 终止其他Python进程
        print("\n=== Cleaning Up Processes ===")
        killed_pids = kill_python_processes()
        if killed_pids:
            print(f"Terminated Python processes: {killed_pids}")
            time.sleep(2)  # 等待进程完全关闭
        else:
            print("No other Python processes found")
        
        # 检查端口状态
        print("\n=== Checking Port Status ===")
        port = 7860
        if is_port_in_use(port):
            print(f"Port {port} is in use, waiting for release...")
            if not wait_for_port_release(port):
                print(f"Port {port} is still in use after timeout")
                print("Please try the following:")
                print("1. Close any running Python processes")
                print("2. Close any web browsers using port 7860")
                print("3. Wait a few minutes and try again")
                sys.exit(1)
        print(f"Port {port} is available")
        
        # 设置环境变量
        print("\n=== Setting Up Environment ===")
        cache_dir = "D:/AI/huggingface"
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
        os.environ['HF_HOME'] = cache_dir
        os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
        print(f"Cache directory: {cache_dir}")
        
        # 确保目录存在
        ensure_directories()
        
        # 创建和启动UI
        print("\n=== Starting UI ===")
        ui = OptimizedUI()
        ui.run()
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        sys.exit(1) 