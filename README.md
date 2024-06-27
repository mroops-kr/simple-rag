# simple-rag

## English

### Overview
* A simple RAG is built using Python.
* The final answer is obtained through file categorization and communication with LLM in a Java-based WAS.
* The purpose of this system is to extract and provide text based on similarity inspection from the uploaded file contents.

### Configuration
* Implemented using MySQL - connection settings need to be changed (.env file).
* Required tables are in the tables.sql file, so they need to be executed to be created.
* The embedding model used is jhgan/ko-sbert-nli from HuggingFaceEmbeddings - a model that supports Korean well.
* Chroma is used for the Vector DB.
* Document summarization is performed simultaneously with file upload; if summarization is not desired, set rag_process_summary='N' in the .env file.
* For summarization, the OPENAI_API_KEY needs to be set in the .env file.

/api/v1/share/~~ : Shared file API
/api/v1/user/~~ : Personal file API

* Author: mroops@naver.com

------------
## Korean

### 개요
* Python 으로 만드는 간단한 RAG를 구축함.
* Java기반 WAS에서 파일에 대한 카테고리 구성 및 LLM과의 통신을 통해 최종 답을 받아옴.
* 본 시스템은 업로드한 파일 내용 중 유사도 검사에 의한 텍스트를 추출하여 제공하는 것을 목적으로 함.

### 설정
* MySql 을 사용하여 구현함 - 연결 설정 정보 변경 필요 (.env 파일)
* 필요 테이블은 tables.sql 파일에 있으니 실행하여 만들어야 함
* 임베딩 모델은 HuggingFaceEmbeddings의 jhgan/ko-sbert-nli 사용 - 한글 지원이 잘되는 모델
* Vector DB 는 크로마 사용
* 파일 업로드와 동시에 문서 요약 진행, 요약 진행을 원하지 않으면 rag_process_summary='N' 로 설정 필요 (.env 파일)
* 요약 진행시 OPENAI_API_KEY 설정 필요 (.env 파일)

/api/v1/share/~~ : 공유 파일 api
/api/v1/user/~~ : 개인 파일 api

* 작성자: mroops@naver.com
