import os
from googleapiclient.discovery import build
from yt_dlp import YoutubeDL
import json

# API 키 설정 - JSON 파일에서 불러옴
with open('../api.json', 'r', encoding='utf8') as f:
    data = json.load(f)
    print(data)


# 현재 작업 디렉토리를 기준으로 절대 경로 생성
CURRENT_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'youtube_captions')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"자막 파일이 저장되는 경로: {OUTPUT_DIR}")

youtube = build(data['YOUTUBE_API_SERVICE_NAME'], data['YOUTUBE_API_VERSION'], developerKey=data['YOUTUBE_API_KEY'])

def get_channel_id(channel_handle):
    """채널 핸들로부터 채널 ID를 가져옵니다."""
    request = youtube.search().list(part='snippet', q=channel_handle, type='channel', maxResults=1)
    response = request.execute()
    print('채널 ID:', response['items'][0]['snippet']['channelId'])
    return response['items'][0]['snippet']['channelId']

def get_video_ids(channel_id):
    """채널 ID로부터 모든 동영상 ID를 가져옵니다."""
    video_ids = []
    next_page_token = None

    while True:
        request = youtube.search().list(
            part='id',
            channelId=channel_id,
            maxResults=50,
            pageToken=next_page_token,
            type='video'
        )
        response = request.execute()

        for item in response['items']:
            video_ids.append(item['id']['videoId'])

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return video_ids

def get_video_title(video_id):
    """동영상 ID로부터 제목을 가져옵니다."""
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()
    return response['items'][0]['snippet']['title']

def clean_caption_text(content):
    """VTT 형식의 자막에서 실제 텍스트만 추출합니다."""
    lines = content.split('\n')
    caption_text = []
    is_text_line = False
    current_text = []
    
    for line in lines:
        # WEBVTT 헤더 스킵
        if 'WEBVTT' in line or 'Kind:' in line or 'Language:' in line:
            continue
            
        # 타임스탬프 줄 확인 (예: 00:00:00.000 --> 00:00:02.000)
        if '-->' in line:
            # 이전 텍스트가 있으면 저장
            if current_text:
                caption_text.append(' '.join(current_text))
                current_text = []
            continue
            
        # 숫자로만 된 줄 (자막 번호) 스킵
        if line.strip().isdigit():
            continue
            
        # 실제 자막 텍스트 처리
        text = line.strip()
        if text and not '-->' in text:
            current_text.append(text)
    
    # 마지막 텍스트 처리
    if current_text:
        caption_text.append(' '.join(current_text))
    
    # 중복 제거 및 정리
    final_text = []
    prev_text = None
    
    for text in caption_text:
        text = text.strip()
        if text and text != prev_text:
            final_text.append(text)
            prev_text = text
    
    return '\n'.join(final_text)

def download_captions(video_id, lang='ko'):
    """yt-dlp를 사용하여 자막을 다운로드합니다."""
    print(f'\n{video_id} 영상 자막 다운로드 시작')
    
    # 임시 파일명 설정
    temp_filename = f'temp_caption_{video_id}'
    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': [lang],
        'outtmpl': temp_filename,
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            # 비디오 정보 추출
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            
            # 가능한 자막 파일 패턴들
            possible_files = [
                f"{temp_filename}.{lang}.vtt",
                f"{temp_filename}.{lang}.srv1",
                f"{temp_filename}.{lang}.srv2",
                f"{temp_filename}.{lang}.srv3",
                f"{temp_filename}.ko.vtt",
                f"{temp_filename}.ko.srv1",
                f"{temp_filename}.ko.srv2",
                f"{temp_filename}.ko.srv3"
            ]
            
            # 자막 파일 찾기
            caption_file = None
            found_content = None
            
            for file_path in possible_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content.strip()) > 0:
                                caption_file = file_path
                                found_content = content
                                print(f"자막 파일 발견 및 내용 확인: {file_path}")
                                break
                    except Exception as e:
                        print(f"파일 읽기 오류 {file_path}: {e}")
                        continue
            
            if found_content:
                # 텍스트 정제
                cleaned_captions = clean_caption_text(found_content)
                
                # 결과 미리보기 출력
                print("\n=== 자막 추출 결과 ===")
                print(f"총 {len(cleaned_captions.split())} 단어 추출됨")
                print("미리보기:")
                preview = cleaned_captions[:300] + "..." if len(cleaned_captions) > 500 else cleaned_captions
                print(preview)
                print("=== 추출 완료 ===\n")
                
                # 임시 파일 삭제
                if caption_file:
                    try:
                        os.remove(caption_file)
                    except:
                        pass
                
                return cleaned_captions
            else:
                print(f"자막 파일을 찾을 수 없음 또는 내용이 비어있음: {video_id}")
                return None
                
    except Exception as e:
        print(f"Error downloading captions for video {video_id}: {e}")
        return None
    finally:
        # 임시 파일 정리
        for file_path in possible_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

def save_to_file(title, video_id, captions):
    """제목과 자막을 파일로 저장하고 내용을 즉시 출력합니다."""
    safe_title = "".join(c if c.isalnum() or c.isspace() else "_" for c in title).strip()
    filepath = os.path.join(OUTPUT_DIR, f"{safe_title}_{video_id}.txt")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # 파일 헤더 정보 추가
        header = f"제목: {title}\n"
        header += f"비디오 ID: {video_id}\n"
        header += f"URL: https://www.youtube.com/watch?v={video_id}\n"
        header += "=" * 50 + "\n\n"
        
        # 헤더와 자막 내용 저장
        f.write(header + captions)
        
        print(f'\n=== {safe_title} 자막 파일 저장 완료 ===')
        print(f'저장 위치: {filepath}')
        print(f'제목: {title}')
        print(f'비디오 ID: {video_id}')
        print(f'URL: https://www.youtube.com/watch?v={video_id}')

def main():
    print("YouTube 채널 자막 추출기 시작")
    print(f"저장 경로: {OUTPUT_DIR}")
    
    channel_id = get_channel_id(CHANNEL_HANDLE)
    video_ids = get_video_ids(channel_id)
    
    total_videos = len(video_ids)
    successful_downloads = 0
    
    print(f"\n총 {total_videos}개의 동영상 처리 시작\n")
    
    for i, video_id in enumerate(video_ids, 1):
        print(f"\n처리 중: {i}/{total_videos} - Video ID: {video_id}")
        try:
            title = get_video_title(video_id)
            captions = download_captions(video_id, lang='ko')
            
            if captions:
                save_to_file(title, video_id, captions)
                successful_downloads += 1
            else:
                print(f"자막을 찾을 수 없음: {video_id} - {title}")
        except Exception as e:
            print(f"비디오 처리 중 오류 발생: {video_id} - {str(e)}")
    
    print(f"\n작업 완료!")
    print(f"총 {total_videos}개의 동영상 중 {successful_downloads}개의 자막 추출 성공")
    print(f"저장 경로: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()