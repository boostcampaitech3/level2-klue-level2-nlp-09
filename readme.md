# P stage(Level 2) - KLUE Data Competition_NLP09 (MnM)
> 문장 내 개체간 관계 추출(Relation Extraction)

# KLUE Relation Extraction Competition - NLP 9조(MnM)
> Naver Boostcamp AI Tech 3rd - Level2 P stage

![public 6th](https://img.shields.io/badge/PUBLIC-6th-red?style=plastic) ![private 4th](https://img.shields.io/badge/PRIVATE-4th-red?style=plastic)


## MnM Team Introduction & Score

### Wrap-up Report 

<a href="https://colorful-bug-b35.notion.site/NLP-9-MnM-Wrap-up-report-6d20d7353b7a4e11befe2096c8246f9e"><img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png" width="50"/></a>
**Click Logo**


### Team Logo  
![Team Logo](https://user-images.githubusercontent.com/46811558/162732068-7389c17e-afd5-48b0-a518-c37226416506.png)

### Public & Private Leaderboard
▶ Public Score  
<img width="889" alt="image" src="https://user-images.githubusercontent.com/33839093/162775160-bb39da33-5239-476e-a6cb-f8edfab95fdc.png"> 

▶ Private Score  
<img width="885" alt="image" src="https://user-images.githubusercontent.com/33839093/162775551-348c90f2-f853-49f3-8181-d90fb7afd4b0.png">

### Members
김태일|문찬국|이재학|하성진|한나연|
:-:|:-:|:-:|:-:|:-:
<img src='https://user-images.githubusercontent.com/46811558/162856318-13a478a3-ad96-4e1f-ad24-3e0a92b81eb7.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/162856364-d71ea54c-31df-433f-8968-93ade6da30b5.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/157460675-9ee90b62-7a39-4542-893d-00eafdb0fd95.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/162856411-70847d72-1dbc-4389-b6e5-bcacba95b2ab.jpg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/46811558/162856463-e10110b7-7e68-4469-9418-6165108a3885.jpg' height=80 width=80px></img>
[detailTales](https://github.com/detailTales)|[nonegom](https://github.com/nonegom)|[wogkr810](https://github.com/wogkr810)|[maxha97](https://github.com/maxha97)|[HanNayeoniee](https://github.com/HanNayeoniee)
gimty97@gmail.com|fksl9959@naver.com |jaehahk810@naver.com|maxha97@naver.com |nayeon2.han@gmail.com  

### Members' Role
| Member | Role | 
| --- | --- |
| 김태일 | input sentence tagging, custom model 제작 ,앙상블 코드 제작, 성능 검증 코드 제작 |
| 문찬국 | Multi-Sentence, Tagging 실험, Refactoring, Sweep구현, 앙상블 실험, 협업툴(GIt) 관리, TestRecording 구현 |
| 이재학 | EDA(중복 및 mislabeling 제거), 평가 metric 분석, 하이퍼 파라미터 튜닝(Ray, Optuna), Random Seed 및 EarlyStopping 구현 |
| 하성진 | Custom Loss 구현 및 실험,  Entity Type Restriction 모델 구현 및 실험, KFold 코드 제작, AMP 적용 |
| 한나연 | EDA, Data Augmentation, Curriculum Learning 적용, confusion matrix 시각화 |
  

## Description
네이버 부스트캠프 AI Tech 3기 P-Stage(Level 2) NLP Competition을 위해 작성된 코드입니다. 해당 Competition에서는 [KLUE RE 데이터셋](https://klue-benchmark.com/tasks/70/overview/description)을 이용해서 **문장 내 개체간 관계 추출**작업을 시행합니다. 문장, 단어에 대한 정보를 통해 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다.  

![description](https://user-images.githubusercontent.com/46811558/162737224-113bf211-e380-4109-9ed4-d511e3d13eba.png)

문장 속에서 단어간에 관계성을 파악하는 것은 의미나 의도를 해석할 때 많은 도움이 되고, 그림의 예시와 같이 요약된 정보를 바탕으로 QA시스템과 같은 여타의 시스템 및 서비스 구성이 가능합니다. 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제로, 자연어처리 응용 프로그램에서 중요한 Task입니다. 이번 대회에서는 문장, 단어에 대한 정보를 통해 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시키고, 이를 통해 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다.


## Usage
### 1) Installation
- requirements.txt 파일을 install 합니다. 
```
pip install -r requirements.txt
```

### 2) Train model
- `config.json`에 설정된 hyperparameter로 모델을 학습시키며, 필요에 따라 config파일을 수정해 train.py를 실행할 수 있습니다. 
```
python train.py
```
### 3) Inference
- 학습된 모델을 추론합니다. default는 `./best_model` 경로로 되어 있습니다.
```
python inference.py
```
- 다음과 같이 저장된 모델의 checkpoint를 사용해 inference 할 수 있습니다. 
```
python inference.py --model_dir=./results/checkpoint-500
```


## Config
train.py와 inference.py에서 동일한 실험 세팅을 위해 `config.json`을 사용할 수 있습니다. 

- `model_name`: "klue/bert-base", "klue/roberta-large" 등 모델 주소를 입력으로 받습니다.
- `filter`: true, false로 입력으로 받으며, true 시 특정 기호들을 제거 (※ 한자는 포함)
- `marking_mode`: dataset 'sentence' 내 sub, obj Entity에 tagging
    - `normal`: not use marking
    - `entity`: [sub] {sub['word']} [/sub]
    - `typed_entity`: <S:{sub['type']}> {sub['word']} </S:{sub['type']}>
    - `typed_entity_punc`: @ * {sub['type']} * {sub['word']} @

- `tokenized_function`: 'multi Sentence'를 만들 때 sub, obj Entity에 tagging
    - `default`: sub['word'] + '[SEP]' + obj['word']
    - `multi`: {sub['word']}[SEP]{obj['word']} 어떤 관계일까?
    - `entity`: [sub]{sub['word']}[/sub] [obj]{obj['word']}[/obj] 어떤 관계일까?
    - `typed_entity`: <S:{sub['type']}> {sub['word']} </S:{sub['type']}> <O:{obj['type]}> {obj['word']} </O:{obj['type]}> 어떤 관계일까?
    - `typed_entity_func`: @ * {sub['type']} * {sub['word']} @ # ^ {obj['type]} ^ {obj['word']} # 어떤 관계일까?

- `loss_name`: `default`(labelsmoother), `f1`, `focal`, `ce`, `weightedce`, `rootweightedce`


## Git Commit Rule
협업을 위해 [Git Commit Rule(깃허브 커밋 룰)](https://underflow101.tistory.com/31)을 참고해 Git Commit Rule을 설정했습니다. 
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

---
## Code Structure
## Wrap Up Report Link
[랩업 리포트 링크](https://colorful-bug-b35.notion.site/NLP-9-MnM-Wrap-up-report-6d20d7353b7a4e11befe2096c8246f9e)
