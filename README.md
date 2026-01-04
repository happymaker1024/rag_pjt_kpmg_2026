# rag_pjt
AI 소믈리에 : langchain rag를 활용한 멀티모달 pjt
<img src="prepare/images/ai_sommelier.png" alt="소개 이미지">

# 설치 라이브러리
```
# langchain rag 관련 lib
pip install -qU python-dotenv

# langchain for opeain 
pip install -Uq langchain langchain-openai 

# langchain of google genai
pip install langchain-google-genai

# vectordb
pip install langchain_pinecone

# LCEL chain 그래프로 시각화 lib
pip install -qU grandalf

- 웹앱 관련 lib
pip install fastapi
pip install "uvicorn[standard]"
```

# 웹앱 실행
```
python app_start.py
```

