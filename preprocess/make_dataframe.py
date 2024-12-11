import os
import pandas as pd

def parse_file_content(file_path):
    """
    텍스트 파일의 내용을 파싱하여 각 요소를 추출하는 함수
    
    Parameters:
    file_path (str): 파일 경로
    
    Returns:
    tuple: (영상제목, 비디오ID, URL, 자막)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # 구분자로 메타데이터와 자막 분리
        metadata, subtitles = content.split('==================================================')
        
        # 메타데이터 파싱
        metadata_lines = metadata.strip().split('\n')
        video_title = metadata_lines[0].replace('제목: ', '').strip()
        video_id = metadata_lines[1].replace('비디오 ID: ', '').strip()
        url = metadata_lines[2].replace('URL: ', '').strip()
        
        # 자막 내용 정리
        subtitles = subtitles.strip()
        
        return video_title, video_id, url, subtitles
        
    except Exception as e:
        print(f"Error parsing file {file_path}: {str(e)}")
        return None, None, None, None

def create_detailed_subtitles_df(folder_path):
    """
    지정된 폴더에서 모든 텍스트 파일을 읽어 상세 데이터프레임으로 변환하는 함수
    
    Parameters:
    folder_path (str): 텍스트 파일들이 있는 폴더 경로
    
    Returns:
    pandas.DataFrame: 상세 정보가 포함된 데이터프레임
    """
    # 결과를 저장할 리스트 초기화
    data = []
    
    # 폴더 내의 모든 텍스트 파일 처리
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            # 파일 내용 파싱
            video_title, video_id, url, subtitles = parse_file_content(file_path)
            
            if video_title is not None:  # 파싱이 성공한 경우만 추가
                data.append({
                    '파일명': filename,
                    '영상제목': video_title,
                    '비디오 ID': video_id,
                    'URL': url,
                    '자막': subtitles,
                    '파일 내용': open(file_path, 'r', encoding='utf-8').read()
                })
    
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    return df

# 사용 예시
if __name__ == "__main__":
    folder_path = "../raw/youtube_captions"  # 폴더 경로
    subtitles_df = create_detailed_subtitles_df(folder_path)
    
    # 결과 확인
    print(f"총 {len(subtitles_df)}개의 파일이 처리되었습니다.")
    print("\n데이터프레임 칼럼:")
    print(subtitles_df.columns.tolist())
    print("\n데이터프레임 미리보기:")
    print(subtitles_df[['파일명', '영상제목', '비디오 ID']].head())
    
    # 필요한 경우 CSV 파일로 저장
    subtitles_df.to_csv('temp_source/youtube_subtitles_detailed.csv', index=False, encoding='utf-8-sig')