# IMBK_Bank_Customer_Churn_ML

## 고객 이탈 분류 ML 및 인사이트 분석
---
### 기간
- 2026년 4월 10일
---
### 기술스택 
- 언어(language) : Python
- 데이터 전처리 : Pandas, NumPy
- 데이터 스케일링 : Scikit-learn
- 머신러닝 모델링 : CatBoost, LightGBM, Scikit-learn
- 하이퍼파라미터 최적화 : Optuna
- 모델 해석(XAI) : SHAP
- 데이터 시각화 : Matplotlib
---
### 데이터 출처
- 캐글 Bank Customer Churn Dataset (row: 10000, col:12)
---
### 데이터 전처리
- 모델의 예측 성능을 극대화하기 위해 데이터 정제 및 피처 엔지니어링을 수행하였습니다. 우선 분석 목적과 무관한 `customer_id` 컬럼을 제거하여 노이즈를 최소화하였으며, 범주형 변수인 `gender`와 `country`는 Label Encoding을 통해 수치형으로 변환하여 모델이 학습 가능한 형태로 가공하였습니다. 데이터 전반에 걸쳐 결측치가 없음을 확인한 후, `balance`와 같이 특정 변수의 수치 범위가 다른 변수들에 비해 과도하게 큰 문제를 해결하기 위해 StandardScaler를 적용하여 모든 피처를 표준 정규 분포 형태로 리스케일링하였습니다. 이를 통해 경사하강법 기반 알고리즘의 수렴 속도를 높이고 특정 변수에 의한 모델 편향을 방지하는 안정적인 데이터셋을 구축하였습니다.
