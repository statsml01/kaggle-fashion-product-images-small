import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from preprop import ProductDataset, TextImageClassifier

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_texts, batch_images, batch_targets in tqdm(test_loader, desc="Evaluating"):
            batch_texts = batch_texts.to(device)
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass
            outputs = model(batch_texts, batch_images)

            # 각 레이블별로 예측 클래스
            preds = torch.stack([torch.argmax(torch.softmax(output, dim=-1), dim=-1) for output in outputs], dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 평가 지표 계산
    accuracies = [
        accuracy_score(all_targets[:, i], all_preds[:, i]) for i in range(all_targets.shape[1])
    ]
    accuracy = np.mean(accuracies)

    f1_scores = [
        f1_score(all_targets[:, i], all_preds[:, i], average="macro")
        for i in range(all_targets.shape[1])
    ]
    precision_scores = [
        precision_score(all_targets[:, i], all_preds[:, i], average="macro")
        for i in range(all_targets.shape[1])
    ]
    recall_scores = [
        recall_score(all_targets[:, i], all_preds[:, i], average="macro")
        for i in range(all_targets.shape[1])
    ]

    # 레이블별 점수 평균
    f1 = np.mean(f1_scores)
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)

    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

def decode_labels(pred_ids, true_ids, label_encoders):
    pred_labels = {col: label_encoders[col].inverse_transform([pred])[0] for col, pred in zip(label_encoders.keys(), pred_ids)}
    true_labels = {col: label_encoders[col].inverse_transform([true])[0] for col, true in zip(label_encoders.keys(), true_ids)}
    return pred_labels, true_labels

def visualize_predictions_with_text(model, test_loader, label_encoders, device, num_samples=5):
    model.eval()
    samples_shown = 0

    with torch.no_grad():
        for batch_texts, batch_images, batch_targets in test_loader:
            batch_texts = batch_texts.to(device)
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass
            outputs = model(batch_texts, batch_images)

            # 각 레이블별로 예측 클래스
            preds = torch.stack([torch.argmax(torch.softmax(output, dim=-1), dim=-1) for output in outputs], dim=1)

            # 데이터를 CPU로 이동
            preds = preds.cpu().numpy()
            batch_targets = batch_targets.cpu().numpy()
            batch_images = batch_images.cpu().numpy()

            for i in range(batch_images.shape[0]):
                if samples_shown >= num_samples:
                    return

                # 텍스트 라벨 복원
                pred_labels, true_labels = decode_labels(preds[i], batch_targets[i], label_encoders)

                # 이미지 준비
                image = batch_images[i].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                image = (image - image.min()) / (image.max() - image.min())  # 정규화

                # 시각화
                plt.figure(figsize=(5, 5))
                plt.imshow(image)
                plt.axis("off")
                plt.title(
                    f"Prediction: {pred_labels}\nTrue Labels: {true_labels}",
                    fontsize=10
                )
                plt.show()

                samples_shown += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model and visualize predictions.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the preprocessed data.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize.")
    args = parser.parse_args()

    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    test_data = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    vocab = torch.load(os.path.join(args.data_dir, "vocab.pth"))
    label_encoders = torch.load(os.path.join(args.data_dir, "label_encoders.pth"))
    test_dataset = ProductDataset(test_data, vocab)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 모델 로드
    target_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
    num_classes_per_label = [test_data[col].nunique() for col in target_columns]
    model = TextImageClassifier(num_classes_per_label).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    # 평가
    evaluate_model(model, test_loader, device)

    # 예측 결과 시각화
    visualize_predictions_with_text(model, test_loader, label_encoders, device, num_samples=args.num_samples)
