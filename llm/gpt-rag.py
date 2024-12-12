# # GPT 버전 사용 시
# from gpt_rag import PerfumeRAG as GPTPerfumeRAG

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
import json

# API 키 설정
with open('../api.json', 'r', encoding='utf8') as f:
    data = json.load(f)


class PerfumeRAG:
    def __init__(self, persist_directory="./chroma_gpt_db", model_name="gpt-4o-mini"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            api_key=data['CHATGPT_API_KEY'],
            model="text-embedding-ada-002"
        )
    
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
                api_key=data['CHATGPT_API_KEY']
            )

    def create_vectorstore(self, df):
        """
        향수 데이터로부터 벡터 스토어 생성
        """
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
        """
        새로운 향수에 대한 설명 생성
        """
        if self.vectorstore is None:
            raise ValueError("Vector store has not been initialized yet")

        # 노트 정보를 기반으로 유사한 향수 검색
        query = f"Notes: {perfume_notes}"
        similar_docs = self.vectorstore.similarity_search(query, k=k)

        # 유사한 향수들의 설명만 추출하여 context 구성
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

위 향료 조합으로 이루어진 향수의 향을 자세하고 시적으로 서너 문장 안으로 설명해주세요. 첫 향이 어떻게 펼쳐지는지, 어떤 감정과 이미지를 불러일으키는지,
어떤 특별한 장면이나 기억이 연상되는지, 향의 전반적인 특성과 개성은 어떠한지, 시간에 따라 어떻게 변화하는지, 이 향만의 독특한 특징은
무엇인지 등을 포함해서 설명해주세요.

참조 향수의 설명은 직접적으로 언급하지 말고, 새로운 향수만의 고유한 설명을 작성해주세요."""

        try:
            response = self.model.predict(prompt)
            return response.strip()

        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            print(f"Prompt: {prompt}")
            raise
        
        
# # 사용 예시
# # 1. 데이터 로드
# df = pd.read_csv("../preprocess/final/rag-gpt.csv")
# print(f"로드된 향수 데이터: {len(df)}개")

# # 2. RAG 시스템 초기화
# rag = GPTPerfumeRAG()

# # 3. 벡터 스토어 생성 (처음 한 번만 실행하면 됨)
# if rag.vectorstore is None:
#     rag.create_vectorstore(df)

# # 4. 설명 생성
# test_notes = "Salt, Vanilla, Watery, Rose"
# description = rag.generate_description(test_notes)
# print("\n생성된 설명:")
# print(description)