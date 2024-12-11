import re
import pandas as pd

def clean_subtitle_text(text):
    """
    자막 텍스트를 전처리하는 함수
    
    Parameters:
    text (str): 원본 자막 텍스트
    
    Returns:
    str: 전처리된 자막 텍스트
    """
    if not isinstance(text, str):
        return ""
        
    # 타임스탬프(<00:00:00.000>) 및 XML 태그(<c>) 제거
    clean_text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', text)
    clean_text = re.sub(r'</?c>', '', clean_text)
    
    # 줄별로 분리
    lines = clean_text.split('\n')
    
    # 중복 제거를 위한 세트
    unique_lines = []
    seen_lines = set()
    
    for line in lines:
        # 줄 앞뒤 공백 제거
        line = line.strip()
        
        # 빈 줄 무시
        if not line:
            continue
            
        # 이미 본 줄이면 건너뛰기
        if line in seen_lines:
            continue
            
        # 새로운 줄 추가
        unique_lines.append(line)
        seen_lines.add(line)
    
    # 정제된 줄들을 하나의 텍스트로 합치기
    return ' '.join(unique_lines)

df = pd.read_csv('temp_source/youtube_subtitles_detailed.csv')

# 자막 열 전처리
print("자막 전처리 중...")
df['전처리된_자막'] = df['자막'].apply(clean_subtitle_text)

# 결과 저장
df.to_csv('temp_source/youtube_subtitles/preprocessed_subtitles.csv', index=False, encoding='utf-8-sig')
