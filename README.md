# Library QoE — 도서관 WiFi 체감 품질 예측

**KT 오픈데이터 챌린지 3위 수상작.** 도서관 WiFi 로그 9,193건으로 체감 품질(QoE) 지표를 직접 설계하고, GRU로 5분 뒤 QoE를 예측해 좌석 선택에 쓰도록 만든 서비스입니다.

## 왜 지표를 새로 만들었나

원본 로그에는 정답 레이블이 없습니다. 핑, 지터, 손실률, 속도, 신호 세기, 접속자 수만 있을 뿐이라 "이 자리가 쾌적한가"를 판단할 기준이 없어서 QoE 지표를 직접 정의했습니다.

- **로그 정규화.** 핑이 10ms에서 40ms로 오를 때의 체감 저하가 300ms에서 330ms로 오를 때보다 큽니다. 선형 정규화는 이 차이를 못 잡아서 `log1p` 스케일을 썼습니다.
- **가중치.** 지연 0.25, 다운로드 0.25, 손실 0.20, 지터 0.15, RSSI 0.10, 접속자 수 0.05. 지연과 손실 계열에 0.6을 몰아 체감 저하를 먼저 반영합니다. RSSI와 접속자 수는 원인 변수에 가까워 비중을 낮췄습니다.
- **평활.** EWM(span=12, 5분 간격 기준 약 1시간)으로 순간 튐을 걸러냅니다. 좌석을 고르는 사람에게 필요한 건 1분짜리 스파이크가 아니라 앞으로 머물 구간의 추세입니다.
- **3단계 변환.** 서비스가 보여주는 값은 연속값이 아니라 Good/Moderate/Bad입니다. 임계값은 학습 구간의 33/66 분위수로 잡았고 `backend/models/thresholds.txt`에 저장돼 있습니다.

정의는 `ml/qoe_features.py` 한 곳에 모아 학습과 서빙이 같은 산식을 쓰도록 했습니다.

## 예측 문제와 기준선

30분치 시퀀스(6스텝)를 입력해 5분 뒤 QoE를 예측합니다. QoE는 자기상관이 강해서 "5분 뒤는 지금과 같다"는 Persistence 기준선이 이미 꽤 잘 맞습니다. 이 선을 넘지 못하면 시계열 모델을 쓸 이유가 없어서 먼저 계산해 두고 비교합니다.

| 기준선 | MSE | RMSE | MAE |
|---|---|---|---|
| Persistence (테스트 1,808스텝) | 0.000229 | 0.015148 | 0.010950 |

```bash
python ml/baseline.py
```

## 모델 비교

GRU, CNN-LSTM, Transformer 인코더, LightGBM을 같은 분할과 같은 지표로 비교했습니다. 최종 서빙 모델은 GRU이고 `backend/models/gru_qoe.h5`로 저장돼 FastAPI가 바로 로드합니다.

```bash
pip install -r ml/requirements.txt
python ml/train_models.py --csv backend/db/dataset_plus.csv --plot
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
