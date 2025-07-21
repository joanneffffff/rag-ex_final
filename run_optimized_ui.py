#!/usr/bin/env python3
"""
Start the integrated RAG UI system - combines multi-stage retrieval and traditional RAG.
Chinese queries: use AlphaFin multi-stage retrieval.
English queries: use traditional RAG system.
"""

import sys
import os
from pathlib import Path
from config.parameters import Config

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Start the integrated RAG UI system"""
    try:
        # Check if gradio is installed
        try:
            import gradio as gr
        except ImportError:
            print("Gradio is not installed, installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
            print("Gradio installation completed")
        
        print("Starting the integrated RAG UI system...")
        print("Access URL: http://localhost:7860")
        print("Press Ctrl+C to stop server")
        
        # Use platform-aware config
        config = Config()
        
        # Import and start the integrated UI
        from xlm.ui.optimized_rag_ui import OptimizedRagUI
        
        # Create UI instance using the integrated version
        ui = OptimizedRagUI(
            cache_dir=config.cache_dir,
            use_faiss=True,
            enable_reranker=True,
            window_title="Financial Explainable RAG System",
            title="Financial Explainable RAG System",
            examples=[
                ["德赛电池（000049）2021年利润持续增长的主要原因是什么？"],
                ["用友网络2019年的每股经营活动产生的现金流量净额是多少？"],
                ["首钢股份的业绩表现如何？"],
                ["钢铁行业发展趋势"],
                ["富春环保（002479）的最新研究报告显示，国资入主对公司有何影响？"],
                ["How was internally developed software capitalised?"], # text query
                ["Which years does the table provide information for net sales by region?"], # table+text query
                ["What was the total cost for 2019?"] # table query
            ]
        )
        
        # Launch the UI
        ui.launch(share=False)
        
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Startup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 