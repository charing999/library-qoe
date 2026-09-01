# Library QoE — 도서관 WiFi 체감 품질 예측

**KT 오픈데이터 챌린지 3위 수상작.** 도서관 WiFi 로그 9,193건으로 체감 품질(QoE) 지표를 직접 설계하고, 5분 뒤 QoE를 예측해 좌석 선택에 쓰도록 만든 서비스입니다. GRU·CNN-LSTM·Transformer·LightGBM을 Persistence 기준선과 함께 비교했고, 그 결과 지표 설계가 예측 문제를 자명하게 만들었다는 점까지 아래에 정리했습니다.

## 왜 지표를 새로 만들었나

원본 로그에는 정답 레이블이 없습니다. 핑, 지터, 손실률, 속도, 신호 세기, 접속자 수만 있을 뿐이라 "이 자리가 쾌적한가"를 판단할 기준이 없어서 QoE 지표를 직접 정의했습니다.

- **로그 정규화.** 핑이 10ms에서 40ms로 오를 때의 체감 저하가 300ms에서 330ms로 오를 때보다 큽니다. 선형 정규화는 이 차이를 못 잡아서 `log1p` 스케일을 썼습니다.
- **가중치.** 지연 0.25, 다운로드 0.25, 손실 0.20, 지터 0.15, RSSI 0.10, 접속자 수 0.05. 지연과 손실 계열에 0.6을 몰아 체감 저하를 먼저 반영합니다. RSSI와 접속자 수는 원인 변수에 가까워 비중을 낮췄습니다.
- **평활.** EWM(span=12, 5분 간격 기준 약 1시간)으로 순간 튐을 걸러냅니다. 좌석을 고르는 사람에게 필요한 건 1분짜리 스파이크가 아니라 앞으로 머물 구간의 추세입니다.
- **3단계 변환.** 서비스가 보여주는 값은 연속값이 아니라 Good/Moderate/Bad입니다. 임계값은 학습 구간의 33/66 분위수로 잡았고 `backend/models/thresholds.txt`에 저장돼 있습니다.

정의는 `ml/qoe_features.py` 한 곳에 모아 학습과 서빙이 같은 산식을 쓰도록 했습니다.

## 예측 문제와 기준선

30분치 시퀀스(6스텝)를 입력해 5분 뒤 QoE를 예측합니다. 그런데 모델을 붙이기 전에 "5분 뒤는 지금과 같다"고만 답하는 Persistence 기준선을 먼저 계산했습니다.

```bash
python ml/baseline.py
```

## 결과: 기준선이 모든 모델을 이겼습니다

테스트 구간 1,808스텝 기준입니다.

| 모델 | MSE | MAE | 3단계 정확도 | F1-macro |
|---|---|---|---|---|
| **Persistence (기준선)** | **0.000229** | **0.010950** | **0.9403** | **0.9278** |
| LightGBM | 0.000542 | 0.017193 | 0.8744 | 0.8546 |
| CNN-LSTM | 0.000544 | 0.017025 | 0.8540 | 0.8212 |
| Transformer | 0.000672 | 0.019065 | 0.8451 | 0.8269 |
| GRU | 0.001069 | 0.024420 | 0.7987 | 0.7738 |

예측 지평을 1시간(12스텝)으로 늘려도 순위는 그대로였습니다. Persistence가 MAE 0.0226, 정확도 0.8499로 여전히 1위입니다.

원인은 QoE 지표 자체에 있습니다. EWM(span=12)으로 평활하면 한 스텝 변화량이 새 관측치의 약 1/6만 반영되어 계열이 거의 임의보행에 가까워집니다. 자기상관이 이렇게 강하면 5분 뒤 값은 현재 값이 최적 추정치에 가깝고, 딥러닝 모델은 그 자명한 해를 학습으로 되찾느라 오히려 손해를 봅니다.

**그래서 배운 것.** 기준선 없이 MSE 0.001만 보고했다면 잘 맞는 모델처럼 보였을 것입니다. 기준선을 계산한 덕분에 문제 설정 자체가 잘못됐다는 걸 알았습니다. 다음에 바꿀 지점은 모델이 아니라 세 가지입니다.

- 평활을 걷어내고 원 QoE를 예측하거나, 평활 강도를 낮춥니다
- 절대값 대신 **변화량**(QoE(t+h) − QoE(t))을 타깃으로 둡니다. 기준선의 예측은 항상 0이 되므로 모델이 실제로 무언가를 더해야 합니다
- 지금은 AP 구분 없이 한 계열로 다루고 있어, 층·구역별로 나눠 공간 정보를 살립니다

`backend/models/gru_qoe.h5`로 서빙 중인 모델은 이 비교 이전에 학습한 것이라 위 표의 GRU와 같은 조건은 아닙니다.

## 모델 비교 실행

GRU, CNN-LSTM, Transformer 인코더, LightGBM을 같은 분할·같은 지표로 비교하고 기준선을 같은 표에 넣습니다.

```bash
pip install -r ml/requirements.txt
python ml/train_models.py --plot                # 5분 뒤 예측
python ml/train_models.py --horizon 12 --plot   # 1시간 뒤 예측
```

- `notebooks/qoe_model_comparison.ipynb` — 지표 설계부터 모델 비교까지 순서대로 따라가는 노트북
- `ml/qoe_features.py` — QoE 산식과 피처 생성
- `ml/baseline.py` — Persistence 기준선
- `ml/train_models.py` — 네 모델 학습·평가·시각화

## 구조

```
backend/    FastAPI 서빙. 학습된 GRU와 scaler를 로드해 층·구역별 QoE를 예측
  db/       수집 로그 (5분 간격, 9,193행)
  models/   gru_qoe.h5, scaler.pkl, thresholds.txt
ml/         모델링 코드
notebooks/  모델 비교 노트북
frontend/   React + Vite. 도서관 도면 위에 예측 QoE를 표시
```

## 실행 방법

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```
