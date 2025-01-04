# 상품 카테고리 분류

상품의 타이틀(`product` 컬럼)과 이미지 특징(`img_feat` 컬럼)을 입력으로 활용하여 세부 카테고리를 예측합니다. 

텍스트 인코더로는 BERT와 LSTM 두 가지 옵션을 제공하며, 이는 각각 다음과 같은 근거로 설계되었습니다:

1. LSTM (기본값):

   - 효율성: 상대적으로 작은 모델로, 학습 및 추론이 빠르며, 제한된 하드웨어에서도 실행 가능합니다.
   - 데이터 특성: 학습 데이터는 비교적 간단한 상품명으로 구성되어 있으며, 문맥 이해가 복잡하지 않아 LSTM의 단순한 구조로도 충분히 학습이 가능합니다.
   - 모델 경량화: 데이터가 오래된 상품 정보를 포함하고 있어, 최신 사전 학습 모델(BERT 등)을 사용하는 것보다 효율적인 학습이 가능합니다.

2. BERT (옵션):

   - 강력한 표현력: 사전 학습된 BERT는 복잡한 문맥 이해가 필요한 데이터에서 우수한 성능을 발휘합니다.
   - 확장 가능성: 특정 상품명에서 의미나 문맥이 중요한 경우, BERT를 선택하여 더 높은 성능을 기대할 수 있습니다.

이미지 인코더로는 ResNet18을 사용하여 이미지를 효율적으로 처리합니다. 이는 다음과 같은 이유로 선택되었습니다:

- 표준적인 성능: ResNet18은 이미지 분류에서 널리 사용되는 모델로, 학습 안정성과 성능이 입증되었습니다.
- 데이터 적합성: 상품 이미지는 일반적으로 객체 탐지가 아닌 기본적인 특징 추출이 요구되므로, ResNet18의 특징 추출 능력으로 충분합니다.
- 경량 모델: ResNet18은 고성능과 효율적인 리소스 사용의 균형을 제공합니다.

### 모델 설계

텍스트와 이미지의 특징 벡터를 결합한 후, 각 타겟 변수에 대해 독립적인 분류기를 적용하여 다중 클래스 분류 문제를 해결합니다. 이 구조는 각 레이블에 대해 독립적으로 최적화할 수 있어 높은 유연성을 제공하며, 텍스트와 이미지 간 상호작용을 효과적으로 학습할 수 있도록 설계되었습니다.

---

## Requirements

- Python 3.10
- PyTorch 2.3

macOS 환경에서 실행을 확인하였으며, CPU 및 GPU 환경 모두에서 동작이 가능합니다.

---

## 데이터

1. 데이터를 다운로드하려면 아래 코드를 실행합니다.

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")

print("Path to dataset files:", path)
```

2. 데이터는 이미지와 함께 텍스트 정보를 포함하며, 상품의 다양한 특징(예: 색상, 카테고리)을 포함합니다. 데이터는 학습과 테스트로 분리하여 사용합니다.

---

## 결과

### 1. 정량 평가

- Accuracy: 0.8682
- F1 Score: 0.5807
- Precision: 0.6171
- Recall: 0.5684

### 2. 정성 평가

다음은 모델이 예측한 카테고리와 실제 라벨을 비교한 예시입니다:

#### 예시 1
![Sample 1](sample_1.png)

#### 예시 2
![Sample 2](sample_2.png)

#### 예시 3
![Sample 3](sample_3.png)

#### 예시 4
![Sample 4](sample_4.png)

#### 예시 5
![Sample 5](sample_5.png)

의견: 모델은 대부분의 카테고리를 정확히 예측하였으나, 일부 예측에서 오차가 발생하였습니다. 예를 들어, '사용 용도(usage)'와 '기본 색상(baseColour)'은 텍스트와 이미지 모두에서 분류가 까다로운 요소로 나타났습니다. 이를 개선하기 위해 추가적인 데이터 증강 또는 더 깊은 모델을 활용할 수 있습니다.

---

## 결과 재현 - 실행 방법

```bash
#!/bin/bash

# 1. 데이터 전처리
python preprocess.py --output_dir data/

# 2. 모델 학습
python train.py --data_dir data/ --checkpoint_file checkpoints/model_checkpoint.pth --num_epochs 3 --batch_size 32 --learning_rate 1e-4

# 3. 평가 및 시각화
python evaluation.py --data_dir data/ --checkpoint_path checkpoints/model_checkpoint.pth --num_samples 5
```

---

모델의 구조를 고려했을 때, LSTM은 짧고 단순한 텍스트 데이터셋에서 효율적으로 작동하며, 제한된 하드웨어 환경에서도 빠르게 학습이 가능합니다. 현재 데이터셋의 특성과 평가 결과를 종합적으로 볼 때, 이는 성능과 효율성 간의 균형을 고려한 적절한 모델 선택이라 할 수 있습니다.
