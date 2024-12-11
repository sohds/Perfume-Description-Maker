import pandas as pd
from openai import OpenAI
import time
from typing import List
import json

with open('../api.json', 'r', encoding='utf8') as f:
    data = json.load(f)
    print(data)


def preprocess_perfume_data(df):
    # 향수_이름 열의 공백 제거
    df['향수_이름'] = df['향수_이름'].str.strip()
    
    # 중복된 향수 이름 중 첫 번째 항목만 유지
    df = df.drop_duplicates(subset=['향수_이름'], keep='first')
    
    return df

def get_english_names_batch(client, perfume_names: List[str], batch_size: int = 20):
    """배치 단위로 향수 이름 처리"""
    all_english_names = []
    
    # 배치 단위로 나누기
    for i in range(0, len(perfume_names), batch_size):
        batch = perfume_names[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {len(perfume_names)//batch_size + 1}")
        
        # 배치의 모든 향수 이름을 하나의 프롬프트로 결합
        names_list = "\n".join([f"{idx+1}. {name}" for idx, name in enumerate(batch)])
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides official English names of perfumes. For each numbered perfume name in the list, respond with the same number followed by its English name. Respond with ONLY the numbers and English names, one per line. Always include the brand/house name in your response."
                    },
                    {
                        "role": "user",
                        "content": f"Provide the English names for these perfumes:\n{names_list}"
                    }
                ],
                temperature=0.3
            )
            
            # 응답 파싱
            result = response.choices[0].message.content.strip().split('\n')
            
            # 번호 제거하고 순수 향수 이름만 추출
            english_names = [name.split('. ', 1)[1].strip() if '. ' in name else name.strip() 
                           for name in result]
            print(english_names)
            all_english_names.extend(english_names)
            
            # API 호출 제한 방지
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # 에러 발생 시 해당 배치의 향수 이름들을 'ERROR' 로 처리
            all_english_names.extend(['ERROR'] * len(batch))
    
    return all_english_names

def process_perfume_names(file_path: str, api_key: str, batch_size: int = 20):
    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=api_key)
    
    # 데이터 읽기
    df = pd.read_csv(file_path)
    
    # 데이터 전처리
    df = preprocess_perfume_data(df)
    
    # 배치 처리로 영어 이름 얻기
    english_names = get_english_names_batch(client, 
                                          df['향수_이름'].tolist(), 
                                          batch_size=batch_size)
    
    # 결과를 데이터프레임에 추가
    df['향수_name'] = english_names
    
    # 결과 저장
    output_path = file_path.rsplit('.', 1)[0] + '_eng.csv'
    df.to_csv(output_path, index=False)
    
    return df


# 사용 예시
api_key = data['CHATGPT_API_KEY']
file_path = "temp_source/only_perfume_name_description.csv"
processed_df = process_perfume_names(file_path, api_key)
