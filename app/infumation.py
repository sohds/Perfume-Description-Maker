import streamlit as st
import pandas as pd
import json
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from openai import OpenAI
import requests
from io import BytesIO
from openai import OpenAI
import time
from PIL import Image
    
# streamlit 웹 배포를 위한 절대경로 포함
def get_absolute_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, relative_path)

logo_path = get_absolute_path('forapp/logo.png')
logo = Image.open(logo_path)

class PerfumeRAGApp:
    def __init__(self, persist_directory=None, model_name="gpt-4o-mini"):
        # persist_directory 경로 설정
        if persist_directory is None:
            persist_directory = get_absolute_path("./chroma_db")
        self.persist_directory = persist_directory
        self.client = OpenAI(api_key=st.secrets["openai"]["api_key"])
        
        # OpenAI 임베딩 설정
        self.embeddings = OpenAIEmbeddings(
            api_key=st.secrets["openai"]["api_key"],
            model="text-embedding-ada-002"
        )
        
        # 벡터 스토어 로드 또는 초기화
        if os.path.exists(persist_directory):
            print("기존 벡터 스토어를 로드합니다...")
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("벡터 스토어가 존재하지 않습니다. 새로 생성합니다.")
            self.vectorstore = None

        # ChatGPT 모델 초기화
        self.model = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            api_key=st.secrets["openai"]["api_key"]
        )

    def create_vectorstore(self, df):
        """향수 데이터로부터 벡터 스토어 생성"""
        print(f"벡터 스토어 생성을 시작합니다. 데이터 개수: {len(df)}")

        documents = []
        for _, row in df.iterrows():
            doc = f"""Perfume: {row['향수_이름']}
Notes: {row['향수_notes']}
Description: {row['향수_설명']}"""
            documents.append(doc)

        self.vectorstore = Chroma.from_texts(
            documents,
            self.embeddings,
            metadatas=[{"source": str(i)} for i in range(len(documents))],
            persist_directory=self.persist_directory
        )

        print("벡터 스토어를 저장합니다...")
        self.vectorstore.persist()
        print("벡터 스토어 생성이 완료되었습니다!")

    def generate_description(self, perfume_notes, k=3):
        """향수 설명 생성"""
        if self.vectorstore is None:
            raise ValueError("Vector store has not been initialized yet")

        query = f"Notes: {perfume_notes}"
        similar_docs = self.vectorstore.similarity_search(query, k=k)

        context_parts = []
        for doc in similar_docs:
            description = doc.page_content.split('Description: ')[-1].strip()
            notes = doc.page_content.split('Notes: ')[1].split('\n')[0].strip()
            context_parts.append(f"노트 구성: {notes}\n설명: {description}")

        context = "\n\n".join(context_parts)

        prompt = f"""당신은 전문적인 향수 리뷰어입니다. 아래 참조 향수들의 설명을 참고하여 새로운 향수를 상세하게 설명해주세요.

[참조 향수 설명]
{context}

[새로운 향수 노트]
{perfume_notes}

위 향료 조합으로 이루어진 향수의 향을 자세하고 시적으로 서너 문장 안에 설명해주세요. 첫 향이 어떻게 펼쳐지는지, 어떤 감정과 이미지를 불러일으키는지, 
시간에 따라 어떻게 변화하는지, 이 향만의 독특한 특징은 무엇인지 등을 설명해주세요.

참조 향수의 설명은 직접적으로 언급하지 말고, 새로운 향수만의 고유한 설명을 작성해주세요."""

        try:
            response = self.model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            st.error(f"설명 생성 중 오류가 발생했습니다: {str(e)}")
            return None

    def generate_image_prompt(self, perfume_description):
        """향수 설명을 기반으로 이미지 생성을 위한 프롬프트 생성"""
        prompt = f"""Create an abstract artistic composition inspired by a perfume with these characteristics:

{perfume_description}

Key requirements:
- Use soft, organic shapes and gentle color transitions
- Create an elegant, minimalist composition
- Focus on natural elements and abstract forms
- Avoid any text or specific branding
- Keep the overall tone peaceful and serene
- Use a light, airy color palette"""

        return prompt

    def generate_image(self, description):
        """DALL-E를 사용하여 이미지 생성"""
        try:
            image_prompt = self.generate_image_prompt(description)
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            return response.data[0].url
        except Exception as e:
            st.error(f"이미지 생성 중 오류가 발생했습니다: {str(e)}")
            return None

def simulate_typing(text, placeholder):
    """텍스트 타이핑 효과 시뮬레이션"""
    import time
    delay = 0.03  # 타이핑 속도 (초)
    
    # 한 글자씩 표시
    current_text = ""
    for char in text:
        current_text += char
        placeholder.markdown(current_text)
        time.sleep(delay)

def main():
    st.set_page_config(page_title="🤖 Infumation", layout="wide")
    st.image(logo)
    st.title("🌸 Infumation: 향수 설명 & 이미지 생성기")
    st.write("향료 조합을 입력하면 AI가 향수 설명과 이미지를 생성해드립니다. 🧙")
    
    # 앱 초기화
    if 'app' not in st.session_state:
        st.session_state.app = PerfumeRAGApp()
        
        # 데이터 로드 및 벡터 스토어 생성
        if st.session_state.app.vectorstore is None:
            with st.spinner("데이터를 준비하고 있습니다..."):
                csv_path = get_absolute_path("../preprocess/final/rag-gpt.csv")
                df = pd.read_csv(csv_path)
                st.session_state.app.create_vectorstore(df)
    
    # 세션 상태 초기화
    if 'description' not in st.session_state:
        st.session_state.description = None
    if 'image_url' not in st.session_state:
        st.session_state.image_url = None
    
    # 사이드바에 입력 폼 배치
    with st.sidebar:
        st.header("✨ 향료 입력")
        notes = st.text_area(
            "향료 조합을 입력하세요 (쉼표로 구분)",
            placeholder="예: Rose Absolute, Jasmine, Vanilla"
        )
        k = st.slider("참조할 유사 향수 개수", min_value=1, max_value=5, value=3)
        generate_button = st.button("설명 생성", type="primary")
    
    # 메인 영역에 두 개의 열 생성
    col1, col2 = st.columns(2)
    
    with col1:
        if generate_button and notes:
            with st.spinner('향수 설명을 생성하고 있습니다...'):
                st.session_state.description = st.session_state.app.generate_description(notes, k=k)
                
        if st.session_state.description:
            st.markdown("### 🌺 생성된 향수 설명")
            # success 컨테이너 안에서 타이핑 효과 구현
            success_container = st.success("", icon="📌")
            description_placeholder = success_container.empty()
            
            # 설명이 처음 생성되었을 때만 타이핑 효과 적용
            if generate_button:
                current_text = ""
                for char in st.session_state.description:
                    current_text += char
                    description_placeholder.markdown(current_text)
                    time.sleep(0.03)
            else:
                description_placeholder.markdown(st.session_state.description)
            
            st.markdown("### ✏️ 설명 수정")
            # 수정 가능한 텍스트 영역
            edited_description = st.text_area(
                "이미지 생성을 위해 설명을 수정할 수 있습니다",
                value=st.session_state.description,
                height=200
            )
            if st.button("🎨 이미지 생성"):
                with st.spinner('이미지를 생성하고 있습니다...'):
                    # 수정된 설명을 기반으로 이미지 생성
                    st.session_state.image_url = st.session_state.app.generate_image(edited_description)
    
    with col2:
        if st.session_state.image_url:
            st.markdown("### 🖼️ 생성된 이미지")
            st.image(st.session_state.image_url, use_container_width=True)
            st.write('사용해주셔서 감사합니다. ❤️')
        elif st.session_state.description:
            st.markdown("### 🖼️ 이미지")
            st.info("왼쪽의 '이미지 생성' 버튼을 클릭하면 이미지가 여기에 표시됩니다.")

    if generate_button and not notes:
        st.warning("향료 조합을 입력해주세요.")

if __name__ == "__main__":
    main()