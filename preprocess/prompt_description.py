import pandas as pd
import openai
import time
from tqdm import tqdm
import json
import os

# API 키 설정 - JSON 파일에서 가져옴
with open('../api.json', 'r', encoding='utf8') as f:
    data = json.load(f)
    print(data)

openai.api_key = data['CHATGPT_API_KEY']

def load_progress(output_path):
    """
    기존 진행상황을 불러오는 함수
    """
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        # 이미 처리된 파일명들의 집합 반환
        return set(existing_df['파일명'].unique())
    return set()

def save_results(results, output_path, mode='a'):
    """
    결과를 CSV 파일로 저장하는 함수
    """
    df = pd.DataFrame(results)
    if mode == 'a' and os.path.exists(output_path):
        # 헤더는 파일이 없을 때만 추가
        df.to_csv(output_path, mode=mode, header=not os.path.exists(output_path), 
                 index=False, encoding='utf-8-sig')
    else:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
def clean_api_response(response):
    """
    API 응답에서 마크다운 포맷을 제거하는 함수
    """
    if response:
        # 마크다운 백틱과 json 텍스트 제거
        response = response.replace('```json', '').replace('```', '').strip()
    return response

def create_prompt(script):
    """
    Create an English prompt for ChatGPT while expecting Korean response
    """
    return f"""Please analyze the following Korean perfume review script and extract ONLY the fragrance descriptions.
Focus exclusively on how each perfume smells, its notes, the feelings it evokes, and sensory experiences.

INCLUDE:
- Scent descriptions
- Notes and accords
- Emotional responses
- Sensory experiences
- Metaphors and comparisons about the smell

EXCLUDE:
- Pricing information
- Size/volume details
- Availability information
- Purchase locations
- Packaging descriptions
- General product information

Script to analyze:
{script}

Respond in this exact JSON format:
{{
    "perfumes": [
        {{
            "perfume_name": "향수 이름 (exactly as mentioned)",
            "description": "향에 대한 설명만 (가격, 용량, 구매처 등 제외)"
        }}
    ]
}}

IMPORTANT:
1. Keep ONLY scent-related descriptions
2. Maintain the original speaking style from the script
3. Exclude perfumes that don't have specific scent descriptions
4. Make sure all quotes are properly closed"""


def get_chatgpt_response(prompt, model="gpt-4o-mini"):
    """
    Call ChatGPT API and get response
    """
    try:
        client = openai.OpenAI(api_key=openai.api_key)  # API 키는 환경 변수로 설정
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant that extracts ONLY the fragrance descriptions from perfume reviews. Focus on scent descriptions, notes, feelings, and sensory experiences. Exclude pricing, sizing, and availability information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000  # 토큰 수 증가
        )
        
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"API 호출 중 에러 발생: {str(e)}")
        return None

def process_perfume_reviews(csv_path, output_path):
    """
    Process perfume reviews and save analysis results
    """
    # 입력 CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    # 이미 처리된 파일 확인
    processed_files = load_progress(output_path)
    print(f"이미 처리된 파일 수: {len(processed_files)}")
    
    # 각 리뷰 처리
    batch_size = 5  # 5개의 리뷰마다 저장
    current_batch = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="리뷰 분석 중"):
        # 이미 처리된 파일은 건너뛰기
        if row['파일명'] in processed_files:
            continue
            
        print(f"\n현재 처리 중인 파일: {row['파일명']}")
        script = row['전처리된_자막']
        
        # ChatGPT에 보낼 프롬프트 생성
        prompt = create_prompt(script)
        
        # API 호출
        response = get_chatgpt_response(prompt)
        
        if response:
            try:
                # JSON 응답 파싱
                analysis = json.loads(clean_api_response(response))
                
                # 향수 설명이 없는 경우 처리
                if not analysis.get('perfumes', []):
                    current_batch.append({
                        '파일명': row['파일명'],
                        '영상제목': row['영상제목'],
                        'URL': row['URL'],
                        '향수_이름': None,
                        '향수_설명': None
                    })
                else:
                    # 각 향수별로 결과 저장
                    for perfume in analysis['perfumes']:
                        current_batch.append({
                            '파일명': row['파일명'],
                            '영상제목': row['영상제목'],
                            'URL': row['URL'],
                            '향수_이름': perfume['perfume_name'],
                            '향수_설명': perfume['description']
                        })
                    
            except json.JSONDecodeError as e:
                print(f"\nJSON 파싱 에러 발생 (파일: {row['파일명']})")
                print(f"에러 메시지: {str(e)}")
                print(f"정제된 응답 내용: {clean_api_response(response)}")
                
                # 에러가 발생해도 None으로 저장
                current_batch.append({
                    '파일명': row['파일명'],
                    '영상제목': row['영상제목'],
                    'URL': row['URL'],
                    '향수_이름': None,
                    '향수_설명': None
                })
            
            except KeyError as e:
                print(f"\n키 에러 발생 (파일: {row['파일명']})")
                print(f"에러 메시지: {str(e)}")
                print(f"응답 내용: {response}")
                
                # 키 에러가 발생해도 None으로 저장
                current_batch.append({
                    '파일명': row['파일명'],
                    '영상제목': row['영상제목'],
                    'URL': row['URL'],
                    '향수_이름': None,
                    '향수_설명': None
                })
        
        # batch_size만큼 처리되었거나 마지막 항목이면 저장
        if len(current_batch) >= batch_size or idx == len(df) - 1:
            if current_batch:  # 배치에 데이터가 있는 경우에만 저장
                save_results(current_batch, output_path)
                print(f"\n{len(current_batch)}개의 결과 저장 완료")
                current_batch = []  # 배치 초기화
        
        # API 호출 제한을 위한 대기
        time.sleep(1)
    
    return pd.read_csv(output_path)

def analyze_results(output_path):
    """
    저장된 결과를 분석하는 함수
    """
    results_df = pd.read_csv(output_path)
    
    print("\n분석 결과 요약:")
    print(f"총 영상 수: {results_df['파일명'].nunique()}")
    
    valid_reviews = results_df[results_df['향수_이름'].notna()]
    print(f"향수가 언급된 영상 수: {valid_reviews['파일명'].nunique()}")
    print(f"총 추출된 향수 수: {len(valid_reviews)}")
    
    if len(valid_reviews) > 0:
        print(f"향수가 언급된 영상당 평균 향수 수: {len(valid_reviews) / valid_reviews['파일명'].nunique():.1f}")
    
    no_perfume_reviews = results_df[results_df['향수_이름'].isna()]
    print(f"향수 설명이 없는 영상 수: {len(no_perfume_reviews)}")
    
    return results_df

# 실행 예시
if __name__ == "__main__":
    input_csv = "temp_source/preprocessed_subtitles.csv"  # 입력 CSV 파일명
    output_csv = "temp_source/perfume_analysis_results.csv"  # 출력 CSV 파일명
    
    # 분석 실행 (기존 진행상황이 있다면 이어서 처리)
    results_df = process_perfume_reviews(input_csv, output_csv)
    
    # 최종 결과 분석
    analyze_results(output_csv)