"""
Streamlit UI for querying unified financial data system
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path to import from xlm
sys.path.append(str(Path(__file__).parent.parent))

from xlm.utils.unified_data_loader import UnifiedDataLoader

def initialize_data_loader():
    """Initialize the data loader with proper configuration"""
    return UnifiedDataLoader(
        data_dir="../data",
        cache_dir="D:/AI/huggingface",
        use_faiss=True,
        batch_size=32
    )

def main():
    st.set_page_config(
        page_title="Financial Data Query System",
        page_icon="💰",
        layout="wide"
    )

    st.title("Financial Data Query System")
    st.markdown("### Search across TatQA and AlphaFin datasets")

    # Initialize session state
    if 'data_loader' not in st.session_state:
        with st.spinner("Initializing data loader..."):
            st.session_state.data_loader = initialize_data_loader()
            st.session_state.data_loader.build_unified_index(save_dir="../data/processed")

    # Sidebar for configuration
    st.sidebar.title("Search Configuration")
    
    data_source = st.sidebar.multiselect(
        "Select Data Sources",
        ["TatQA", "AlphaFin", "All"],
        default=["All"]
    )
    
    top_k = st.sidebar.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=3
    )

    # Main search interface
    query = st.text_input(
        "Enter your query",
        placeholder="e.g., What was the revenue in Q4? or 如何分析市场趋势?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("Search")
    with col2:
        if st.session_state.data_loader:
            st.success("System ready!")

    if query and search_button:
        with st.spinner("Searching..."):
            results, scores = st.session_state.data_loader.retriever.retrieve(
                text=query,
                top_k=top_k,
                return_scores=True
            )

            # Filter results based on selected data sources
            if "All" not in data_source:
                filtered_results = []
                filtered_scores = []
                for doc, score in zip(results, scores):
                    source = doc.metadata.source
                    if ("TatQA" in data_source and "tatqa" in source.lower()) or \
                       ("AlphaFin" in data_source and "alphafin" in source.lower()):
                        filtered_results.append(doc)
                        filtered_scores.append(score)
                results = filtered_results
                scores = filtered_scores

            # Display results
            if results:
                for i, (doc, score) in enumerate(zip(results, scores)):
                    with st.expander(f"Result {i+1} (Relevance: {score:.4f})"):
                        st.markdown(f"**Source**: {doc.metadata.source}")
                        st.markdown(f"**Content**:")
                        st.markdown(doc.content)
                        
                        # Display metadata
                        st.markdown("**Metadata**:")
                        for key, value in doc.metadata.__dict__.items():
                            if key != "source":
                                st.markdown(f"- {key}: {value}")
            else:
                st.warning("No results found.")

    # Display some example queries
    with st.sidebar.expander("Example Queries"):
        st.markdown("""
        **TatQA Examples**:
        - What was the revenue in Q4?
        - How is the profit margin calculated?
        - Show me the balance sheet figures
        
        **AlphaFin Examples**:
        - 如何分析股票市盈率
        - 市场趋势分析方法
        - 金融风险评估指标
        
        **Cross-domain Examples**:
        - How to evaluate company performance?
        - What are key financial metrics?
        - Market analysis methodology
        """)

if __name__ == "__main__":
    main() 