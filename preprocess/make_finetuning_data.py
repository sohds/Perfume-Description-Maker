import pandas as pd
import json

data = pd.read_csv('final/final_notes_done.csv', encoding='utf-8')

# Define functions to format the data
def only_notes(row):
    prompt = f"향수 노트: {row['향수_notes']}\n\n해당 향기에 대한 설명을 작성해주세요.\n\n"
    target = row['향수_설명']
    return {"type": "only_notes", "prompt": prompt, "completion": target}

def full_data(row):
    prompt = f"향수 이름: {row['향수_name']}\n향수 노트: {row['향수_notes']}\n\n해당 향수의 향기에 대한 설명을 작성해주세요.\n\n"
    target = row['향수_설명']
    return {"type": "full_data", "prompt": prompt, "completion": target}

def only_names(row):
    prompt = f"향수 이름: {row['향수_name']}\n\n이 향수의 향기에 대한 설명을 작성해주세요.\n\n"
    target = row['향수_설명']
    return {"type": "only_names", "prompt": prompt, "completion": target}

# Define a mapping of formatting functions
formatting_functions = {
    "only_notes": only_notes,
    "full_data": full_data,
    "only_names": only_names
}

# Combine results from all formatting functions
combined_data = []
for name, func in formatting_functions.items():
    try:
        # Apply the formatting function
        formatted_data = data.apply(func, axis=1).tolist()
        combined_data.extend(formatted_data)
    except Exception as e:
        raise RuntimeError(f"{name} 포맷팅 중 문제가 발생했습니다: {e}")

# Save the combined data as a single JSONL file
output_path = 'final/fine_tune_dataset.jsonl'
try:
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in combined_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(output_path, 'SAVED.')
except Exception as e:
    raise RuntimeError(f"파일 저장 중 문제가 발생했습니다: {e}")