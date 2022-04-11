# P stage(Level 2) - KLUE Data Competition_NLP09 (MnM)
> 문장 내 개체간 관계 추출(Relation Extraction)

![public 6th](https://img.shields.io/badge/PUBLIC-6th-red?style=plastic) ![private 4th](https://img.shields.io/badge/PRIVATE-4th-red?style=plastic)

## Team Introduction & Score

### **WrapUp Report**  

<a href="https://colorful-bug-b35.notion.site/NLP-9-MnM-Wrap-up-report-6d20d7353b7a4e11befe2096c8246f9e"><img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png" height=80 width=80px/></a>
**Click Logo**


### **Team Logo**  

![Team Logo](https://user-images.githubusercontent.com/46811558/162732068-7389c17e-afd5-48b0-a518-c37226416506.png)

### **Public & Private LB**
▶ Public Score  
![Public Score](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d7e276c3-20ac-45f9-ae99-2b2960c4da04/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220411%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220411T114401Z&X-Amz-Expires=86400&X-Amz-Signature=6d9c7b8c1f4c26e5bd0abc579d662194b57943f87eb44667e0e36f995cf8d42b&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)  
▶ Private Score  
![Private Score](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c241d95b-aa67-4705-9fbd-f5d0e9717317/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220411%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220411T114433Z&X-Amz-Expires=86400&X-Amz-Signature=2c0752b2733d3578d64da7676badb4e89b9619d48a0a5fd9bbc483954008c659&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

### Members
김태일|문찬국|이재학|하성진|한나연|
:-:|:-:|:-:|:-:|:-:
<img src='https://user-images.githubusercontent.com/46811558/157460675-9ee90b62-7a39-4542-893d-00eafdb0fd95.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/157460675-9ee90b62-7a39-4542-893d-00eafdb0fd95.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/157460675-9ee90b62-7a39-4542-893d-00eafdb0fd95.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/157460675-9ee90b62-7a39-4542-893d-00eafdb0fd95.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/157460675-9ee90b62-7a39-4542-893d-00eafdb0fd95.jpg' height=80 width=80px></img>
[Github](https://github.com/detailTales)|[Github](https://github.com/nonegom)|[Github](https://github.com/wogkr810)|[Github](https://github.com/maxha97)|[Github](https://github.com/HanNayeoniee)
gimty97@gmail.com|fksl9959@naver.com |jaehahk810@naver.com|maxha97@naver.com |nayeon2.han@gmail.com  

### Members' Role
| 팀원 | 역할 | 
| --- | --- |
| 김태일_T3063 | input sentence tagging, custom model 제작 ,앙상블 코드 제작, 성능 검증 코드 제작 |
| 문찬국_T3076 | Multi-Sentence, Tagging 실험, Refactoring, Sweep구현, 앙상블 실험, 협업툴(GIt) 관리, TestRecording 구현 |
| 이재학_T3161 | EDA(중복 및 mislabeling 제거), 평가 metric 분석, 하이퍼 파라미터 튜닝(Ray,Optuna), Random Seed 및 EarlyStopping 구현 |
| 하성진_T3230 | Custom Loss 구현 및 실험,  Entity Type Restriction 모델 구현 및 실험, KFold 코드 제작, AMP 적용 |
| 한나연_T3250 | EDA, Data Augmentation, Curriculum Learning 적용, confusion matrix 시각화 |
  

## Description
네이버 부스트캠프 AITech 3기 P-Stage(Level 2) NLP Competition을 위해 작성된 코드입니다. 해당 Competition에서는 KLUE 데이터셋을 이용해서 '문장 내 개체간 관계 추출'작업을 시행합니다. 문장, 단어에 대한 정보를 통해, 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다.  

![description](https://user-images.githubusercontent.com/46811558/162737224-113bf211-e380-4109-9ed4-d511e3d13eba.png)

문장 속에서 단어간에 관계성을 파악하는 것은 의미나 의도를 해석할 때 많은 도움이 되고, 그림의 예시와 같이 요약된 정보를 바탕으로 QA시스템과 같은 여타의 시스템 및 서비스 구성이 가능합니다. 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제로, 자연어처리 응용 프로그램에서 중요한 Task입니다. 이번 대회에서는 문장, 단어에 대한 정보를 통해 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시키고, 이를 통해 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다.


## Git Commit Rule
협업을 위한 Git Commit Rule을 설정했습니다. Commit Rule설정 및 사용 방법에 대해서는 다음 블로그 [Git Commit Rule (깃허브 커밋 룰)](https://underflow101.tistory.com/31)를 참고했습니다. 
ex) `feat: add new code`

```
- feat      : 새로운 기능 추가
- debug     : 버그 수정
- docs      : 문서 수정
- style     : 코드 formatting, 세미콜론(;) 누락, 코드 변경이 없는 경우
- refactor  : 코드 리팩토링
- test      : 테스트 코드, 리팩토링 테스트 코드 추가
- chore     : 빌드 업무 수정, 패키지 매니저 수정
- exp       : 실험 진행
- merge     : 코드 합칠 경우
- anno      : 주석 작업
- etc       : 기타
```

## Usage
**1) Install**
- requirements.txt 파일을 install 합니다. 
```
pip install -r requirements.txt
```

**2) Training**
- 기본적으로 설정된 hyperparameter로 train.py를 실행할 수 있습니다.
- Train.py 및 Inference.py에서 동일한 Parameter를 사용할 경우 **config.json**을 통해 파라미터를 변경합니다. 
```
python train.py
```
**3) Inference**
- 학습된 모델을 추론합니다. default는 `./best_model` 경로로 되어 있습니다.
```
python inference.py
```
- 저장된 모델의 checkpoint를 변경해서 inference하고 싶다면 다음과 같이 사용이 가능합니다. 
```
python inference.py --model_dir=./results/checkpoint-500
```
