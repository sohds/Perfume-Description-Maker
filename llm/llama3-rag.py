# # Llama 버전 사용 시
# from llama_rag import PerfumeRAG as LlamaPerfumeRAG

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch
import os
import json

# API 키 설정
with open('../api.json', 'r', encoding='utf8') as f:
    data = json.load(f)

class PerfumeRAG:
    def __init__(self, persist_directory="./chroma_db",
                 model_path="./perfume_description_generator/checkpoint-656",
                 device="cuda"):
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

        # 수정된 프롬프트 템플릿
        self.template = """아래 참조 향수들의 설명을 바탕으로, 새로운 향수의 향을 설명해주세요.

[참조 향수 설명]
{context}

[새로운 향수 노트]
{notes}

위 향료 조합으로 이루어진 향수의 향을 다음 관점들을 고려하여 설명해주세요:
- 첫 향이 어떻게 펼쳐지는지
- 이 향이 불러일으키는 감정과 이미지
- 연상되는 특별한 장면이나 기억
- 향의 전반적인 특성과 개성
- 시간에 따른 향의 변화
- 이 향만의 독특하거나 특징적인 면

전문적인 향수 리뷰어의 관점에서 자세하고 시적으로 설명해주세요. 참조 향수의 설명은 직접적으로 언급하지 말고, 새로운 향수만의 고유한 설명을 작성해주세요. 작성했던 말은 절대 반복하지 않습니다."""

        self.prompt = PromptTemplate(
            input_variables=["context", "notes"],
            template=self.template
        )

        self.model = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=model_path,
                torch_dtype=torch.float16,
                device=device,
                max_length=512,
                pad_token_id=0,
                eos_token_id=2
            )
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

        prompt = f"""아래 참조 향수들의 설명을 참고하여 새로운 향수를 설명해주세요:

    [참조 향수 설명]
    {context}

    [새로운 향수 노트]
    {perfume_notes}

    위 향료로 이루어진 향수의 향을 설명해주세요. 참조 향수의 설명은 언급하지 말고, 새로운 향수만의 고유한 설명을 작성해주세요.
    """

        try:
            response = self.model(
                prompt,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                return_full_text=False  # 프롬프트를 제외한 생성된 텍스트만 반환
            )

            if isinstance(response, list) and len(response) > 0:
                if isinstance(response[0], dict) and 'generated_text' in response[0]:
                    return response[0]['generated_text'].strip()
                return response[0].strip()
            else:
                print(f"Unexpected model output format: {type(response)}")
                return str(response)

        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            print(f"Prompt: {prompt}")
            raise
        
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['향수_이름', '향수_notes', '향수_설명']].dropna()
    return df


# # 사용 예시
# # 데이터 로드
# df = load_and_process_data("../preprocess/final/description-rag-finetune.csv")
# print(f"로드된 향수 데이터: {len(df)}개")

# # RAG 시스템 초기화
# rag = LlamaPerfumeRAG(persist_directory="./chroma_db")

# # 벡터 스토어 생성
# if rag.vectorstore is None:
#      rag.create_vectorstore(df)

# # 테스트
# test_notes = "Salt, Vanilla, Watery, Rose"
# description = rag.generate_description(test_notes)
# print("\n생성된 설명:")
# print(description)