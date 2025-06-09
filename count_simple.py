import json
from pathlib import Path

def count_questions():
    # TatQA
    tatqa_files = [
        "data/tatqa_dataset_raw/tatqa_dataset_train.json",
        "data/tatqa_dataset_raw/tatqa_dataset_dev.json",
        "data/tatqa_dataset_raw/tatqa_dataset_test.json"
    ]
    
    print("\n=== TatQA Dataset ===")
    total_tatqa = 0
    for file in tatqa_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Count questions in each item
                count = 0
                for item in data:
                    if isinstance(item, dict) and 'questions' in item:
                        count += len(item['questions'])
                print(f"{Path(file).name}: {count} questions")
                total_tatqa += count
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    print(f"Total TatQA questions: {total_tatqa}")
    
    # AlphaFin
    print("\n=== AlphaFin Dataset ===")
    try:
        dataset_path = "data/alphafin/data.json"
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            train = sum(1 for item in data if item.get('split') == 'train')
            test = sum(1 for item in data if item.get('split') == 'test')
            print(f"Train: {train}")
            print(f"Test: {test}")
            print(f"Total: {len(data)}")
    except FileNotFoundError:
        print("AlphaFin data file not found, loading from HuggingFace...")
        from datasets import load_dataset
        dataset = load_dataset("C1em/alphafin")
        train = len(dataset['train'])
        test = len(dataset['test'])
        print(f"Train: {train}")
        print(f"Test: {test}")
        print(f"Total: {train + test}")

if __name__ == "__main__":
    count_questions() 