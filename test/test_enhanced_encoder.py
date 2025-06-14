"""
Test script for enhanced multimodal encoder
"""

import json
import torch
import numpy as np
from pathlib import Path
from xlm.components.encoder.multimodal_encoder import MultiModalEncoder
from config.parameters import Config, EncoderConfig, RetrieverConfig, ModalityConfig
import re

def extract_time_series_from_text(text):
    """Extract time series dict from text if present, else return None"""
    match = re.search(r'\{.*\}', text)
    if match:
        try:
            data = json.loads(match.group().replace("'", '"'))
            if all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in data.items()):
                return data
        except:
            pass
    return None

def load_sample_data():
    """Load sample data from TatQA and AlphaFin"""
    tatqa_path = Path("data/tatqa_dataset_raw/tatqa_dataset_train.json")
    alphafin_path = Path("data/alphafin/sample_data.json")
    tatqa_samples = []
    alphafin_samples = []
    if tatqa_path.exists():
        with open(tatqa_path, 'r', encoding='utf-8') as f:
            tatqa_samples = json.load(f)[:2]  # 取前2条
    if alphafin_path.exists():
        with open(alphafin_path, 'r', encoding='utf-8') as f:
            alphafin_samples = json.load(f)[:2]  # 取前2条
    return tatqa_samples, alphafin_samples

def build_test_batches(tatqa_samples, alphafin_samples, use_enhanced):
    """构造测试批次，分别适配增强/基础编码器"""
    # TatQA: 只用表格和文本
    tatqa_tables = []
    tatqa_texts = []
    for item in tatqa_samples:
        if 'table' in item:
            tatqa_tables.append(item['table'])
        if 'paragraphs' in item:
            tatqa_texts.extend(item['paragraphs'])
    # AlphaFin: input 里可能有 time-series
    alphafin_texts = []
    alphafin_tables = []
    alphafin_time_series = []
    for item in alphafin_samples:
        inp = item.get('input', '')
        if isinstance(inp, list):
            for subinp in inp:
                ts = extract_time_series_from_text(subinp)
                if ts:
                    alphafin_time_series.append(ts)
                else:
                    alphafin_texts.append(subinp)
        else:
            ts = extract_time_series_from_text(inp)
            if ts:
                alphafin_time_series.append(ts)
            else:
                alphafin_texts.append(inp)
    # 基础编码器：time_series 作为 text
    if not use_enhanced:
        alphafin_texts += [str(ts) for ts in alphafin_time_series]
        alphafin_time_series = []
    return {
        'tatqa': {'text': tatqa_texts, 'table': tatqa_tables},
        'alphafin': {'text': alphafin_texts, 'table': [], 'time_series': alphafin_time_series}
    }

def test_enhanced_encoder():
    print("\n=== Testing Enhanced Multimodal Encoder (TatQA & AlphaFin) ===")
    tatqa_samples, alphafin_samples = load_sample_data()
    config = Config(
        encoder=EncoderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            batch_size=8
        ),
        retriever=RetrieverConfig(num_threads=2),
        modality=ModalityConfig(
            combine_method="weighted_sum",
            text_weight=0.4,
            table_weight=0.3,
            time_series_weight=0.3
        )
    )
    for use_enhanced in [False, True]:
        print(f"\nTesting {'Enhanced' if use_enhanced else 'Basic'} Encoder:")
        batches = build_test_batches(tatqa_samples, alphafin_samples, use_enhanced)
        encoder = MultiModalEncoder(
            config=config,
            use_enhanced_encoders=use_enhanced
        )
        # TatQA
        print("\n[TatQA Sample]")
        try:
            tatqa_emb = encoder.encode_batch(
                texts=batches['tatqa']['text'],
                tables=batches['tatqa']['table']
            )
            print("TatQA Embedding shapes:", {k: v.shape for k, v in tatqa_emb.items()})
        except Exception as e:
            print("TatQA encoding error:", e)
        # AlphaFin
        print("\n[AlphaFin Sample]")
        try:
            alphafin_emb = encoder.encode_batch(
                texts=batches['alphafin']['text'],
                tables=batches['alphafin']['table'],
                time_series=batches['alphafin']['time_series'] if use_enhanced else None
            )
            print("AlphaFin Embedding shapes:", {k: v.shape for k, v in alphafin_emb.items()})
        except Exception as e:
            print("AlphaFin encoding error:", e)

if __name__ == "__main__":
    test_enhanced_encoder() 