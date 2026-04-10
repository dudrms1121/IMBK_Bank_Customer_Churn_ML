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
---
### EDA 및 해석
<img width="560" height="427" alt="image" src="https://github.com/user-attachments/assets/48717dcf-f18c-488f-9d7e-aee2b316a45b" />

#### 보유 상품 수 분포 분석

- 해석
1. 전체 고객의 약 95% 이상이 1개 또는 2개의 상품만을 보유하고 있습니다.
2. 상품을 3개 이상 보유한 다건 가입자의 비중은 매우 낮으며, 4개 이상 보유자는 극소수에 불과합니다.

- 비즈니스 인사이트
1. 대다수의 고객이 단일 혹은 소수의 상품만을 이용하는 '단순 거래 고객' 단계에 머물러 있어, 타사로의 이탈 가능성이 높은 상태입니다.
2. 교차 판매 전략의 부재: 고객당 보유 상품 수를 늘려 이탈 방어막을 형성해야 합니다.

<img width="552" height="437" alt="image" src="https://github.com/user-attachments/assets/4b7c5a08-8121-48e1-98b2-5f26e8b2c7f4" />

#### 고객 연령대 분포 분석
- 해석
1. 데이터 내 고객 연령층은 30대 초반에서 40대 초반에 가장 밀집되어 있는 '항아리형' 분포를 보입니다.
2. 20대 초반부터 급격히 유입이 증가하다가, 40대 중반을 기점으로 완만하게 감소하는 추세를 보입니다.

- 비즈니스 인사이트
1. 현재 우리 은행의 주력 고객층은 경제 활동이 가장 활발한 3040 세대임을 알 수 있습니다.
2. 20대 초반의 가파른 상승 곡선은 잠재 고객 확보 가능성을 시사하므로, 이들을 주거래 고객으로 안착시키기 위한 '생애 첫 금융 상품' 등의 타겟 마케팅이 유효할 것으로 판단됩니다.
---
### AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value

##### 1. AutoML & Model Selection

- 다양한 머신러닝 알고리즘의 베이스라인 성능을 비교하여 최적의 상위 모델을 선정하였습니다.

<img width="980" height="559" alt="image" src="https://github.com/user-attachments/assets/8964e7be-adce-4f09-aa22-eb195f67087d" />

- 결과: CatBoost, LightGBM과 GBC가 F1-Score 약 0.59 대를 기록하며 금융 데이터의 불균형 속에서도 안정적인 성능을 보임을 확인했습니다.



##### 2. Hyperparameter Tuning (Optuna)

- 선정된 상위 3개 모델(CatBoost, LGBM, GBC)에 대해 Optuna 프레임워크를 적용, 베이지안 최적화 기반의 하이퍼파라미터 튜닝을 수행했습니다.

- 목적: 각 모델의 오버피팅을 방지하고 F1-Score를 극대화.

- 전략: 10~50회 이상의 Trial을 통해 learning_rate, depth, iterations 등의 최적 조합을 도출했습니다.

##### 3. Stacking Ensemble Pipeline

- 단일 모델의 한계를 극복하기 위해 StackingClassifier를 구축하여 예측력을 한 단계 높였습니다.

- Layer 1 (Base Estimators): Optuna로 최적화된 CatBoost, LGBM, GBC

- Layer 2 (Final Estimator): Logistic Regression을 메타 모델로 사용하여 각 모델의 예측 결과를 최종 통합.

- 결과: 단일 모델 대비 더욱 견고한 예측 성능 확보.

##### 4. Model Interpretation (SHAP Value)

- 모델의 판단 근거를 시각화하기 위해 SHAP 분석을 수행했습니다.

<img width="757" height="550" alt="image" src="https://github.com/user-attachments/assets/88b460c4-1181-47a1-9a13-b4c761f68c64" />


#### 핵심 변수 기여도: Age와 Products_Number가 이탈 예측에 가장 결정적인 역할을 수행.
---
### 인사이트 및 제언

##### 1. 연령대별 차별화된 리텐션 전략의 필요성

- 현상: SHAP 분석 결과 이탈에 가장 지대한 영향을 미치는 요인은 나이로 나타났습니다. 특히 은퇴 전후 시점의 고객층에서 이탈 징후가 뚜렷합니다.

- 해석: 이는 기존 서비스가 고령층 고객의 변화된 금융 니즈(자산 인출, 상속 등)를 충실히 반영하지 못하고 있음을 시사합니다.

- 제언: 중장년층을 위한 전용 자산 관리 솔루션과 시니어 친화적 디지털 뱅킹 UI/UX 개선을 통해 고객 이탈을 선제적으로 방어해야 합니다.

##### 2. 양적 팽창에서 질적 관리로의 패러다임 전환

- 현상: 상품 보유 수와 고객 충성도는 단순 비례 관계가 아님이 확인되었습니다. 오히려 다수의 상품을 보유한 고객이 관리 미흡 시 더 큰 기회비용을 느끼고 이탈할 위험이 존재합니다.

- 해석: 무분별한 상품 가입 권유는 고객에게 관리 피로도를 유발하며, 실제 혜택 체감도를 떨어뜨릴 수 있습니다.

- 제언: 상품 가입 '개수' 중심의 마케팅에서 벗어나, 주거래 고객이 확실한 우대 혜택을 체감할 수 있는 패키지 리워드 체계 및 주거래 고객 우대 제도의 실효성을 재점검해야 합니다.
