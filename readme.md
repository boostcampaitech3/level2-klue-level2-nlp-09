# KLUE Data Competition_NLP09 (MnM)
## Description
네이버 부스트캠프 AITech 3기 Level2_PStage NLP Competition을 위해 작성된 코드입니다. 해당 Competition에서는 KLUE 데이터셋을 이용해서 '문장 내 개체간 관계 추출'작업을 시행합니다. 문장, 단어에 대한 정보를 통해, 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다. 

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