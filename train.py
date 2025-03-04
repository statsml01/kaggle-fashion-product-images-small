from functools import partial
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

from transformers import BertModel, BertTokenizer
from torchvision.models import resnet18

from torchtext.data.utils import get_tokenizer
from preprocess import text_pipeline, preprocess_image


class ProductDataset(Dataset):
    def __init__(self, data, text_pipeline, image_transform):
        self.text_pipeline = text_pipeline
        self.image_transform = image_transform

        # 이미지 경로가 유효한 데이터만 필터링
        valid_indices = []
        for idx, row in data.iterrows():
            image_path = row['image_path']
            if os.path.exists(image_path):  # 이미지 파일 존재 확인
                valid_indices.append(idx)

        # 유효한 데이터만 저장
        self.data = data.loc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 텍스트 처리
        text = self.data.iloc[idx]['productDisplayName']
        text_tensor = self.text_pipeline(text)

        # 이미지 처리
        image_path = self.data.iloc[idx]['image_path']
        image_tensor = self.image_transform(image_path)

        # 타겟 라벨
        targets = self.data.iloc[idx][['gender', 'masterCategory', 'subCategory',
                                       'articleType', 'baseColour', 'season', 'usage']].values.astype(int)
        target_tensor = torch.tensor(targets, dtype=torch.long)

        return text_tensor, image_tensor, target_tensor


class TextImageClassifier(nn.Module):
    def __init__(self, num_classes_per_label, num_labels=7, use_bert=False):
        super(TextImageClassifier, self).__init__()

        self.use_bert = use_bert
        self.num_labels = num_labels
        # 1. 텍스트 인코더
        if use_bert:
            # BERT 기반 텍스트 인코더
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
            self.text_hidden_size = 768  # BERT hidden size
        else:
            # LSTM 기반 텍스트 인코더
            self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=128, padding_idx=0)
            self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
            self.text_hidden_size = 256 * 2  # Bidirectional LSTM

        # 2. 이미지 인코더 (ResNet18)
        resnet = resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(
            *(list(resnet.children())[:-1]),  # 마지막 FC Layer 제거
            nn.Flatten()
        )
        self.image_hidden_size = resnet.fc.in_features  # ResNet18의 feature 크기

        # 레이블별 분류기
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.text_hidden_size + self.image_hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)  # 각 레이블의 클래스 개수
            )
            for num_classes in num_classes_per_label
        ])

    def forward(self, text_input, image_input):
        # 텍스트 인코딩
        if self.use_bert:
            # BERT 인코더
            tokens = self.tokenizer(text_input, padding=True, truncation=True, max_length=50, return_tensors="pt")
            bert_output = self.text_encoder(**tokens)
            text_features = bert_output.pooler_output  # [CLS] 토큰의 출력
        else:
            # LSTM 인코더
            embedded_text = self.embedding(text_input)
            lstm_output, _ = self.lstm(embedded_text)
            text_features = lstm_output[:, -1, :]  # 마지막 Hidden State

        image_features = self.image_encoder(image_input)

        combined_features = torch.cat((text_features, image_features), dim=1)

        outputs = [classifier(combined_features) for classifier in self.classifiers]
        return outputs


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch+1}")

# 텍스트 데이터 처리 함수


# 패딩 적용


def pad_sequence(sequences, max_len=50):
    padded_sequences = torch.zeros((len(sequences), max_len), dtype=torch.int64)
    for i, seq in enumerate(sequences):
        seq = seq[:max_len]  # Max Length 자르기
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences


def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint


# Collate Function: 배치 내 패딩 처리


def collate_fn(batch):
    texts, images, targets = zip(*batch)
    texts = pad_sequence(texts)  # 텍스트 패딩
    images = torch.stack(images)  # 이미지 배치화
    targets = torch.stack(targets)  # 타겟 배치화
    return texts, images, targets


def train_model(data_dir, checkpoint_file, num_epochs=3, batch_size=32, learning_rate=1e-4):
    # Device 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 데이터 로드
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    vocab = torch.load(os.path.join(data_dir, "vocab.pth"))
    # 토큰화 및 Vocabulary 생성
    tokenizer = get_tokenizer("basic_english")

    train_dataset = ProductDataset(train_data, partial(
        text_pipeline, vocab=vocab, tokenizer=tokenizer), preprocess_image)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 모델 초기화
    target_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
    num_classes_per_label = torch.load(os.path.join(args.data_dir, "num_classes_per_label.pth"))
    model = TextImageClassifier(num_labels=len(target_columns),
                                num_classes_per_label=num_classes_per_label, use_bert=False).to(device)

    # 손실 함수 및 옵티마이저 설정
    criteria = [nn.CrossEntropyLoss() for _ in range(len(num_classes_per_label))]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 기존 체크포인트 로드
    start_epoch = 0
    best_loss = float("inf")
    if os.path.exists(checkpoint_file):
        checkpoint = load_checkpoint(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        print(f"Resuming from checkpoint: epoch {start_epoch}, loss {best_loss:.4f}")

    # 학습 루프
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_texts, batch_images, batch_targets in progress_bar:
            # 데이터 전송
            batch_texts = batch_texts.to(device)
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(batch_texts, batch_images)

            # 레이블별 손실 계산
            loss = 0
            for i in range(len(num_classes_per_label)):
                loss += criteria[i](outputs[i], batch_targets[:, i])

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{running_loss/len(progress_bar):.4f}"})

        # 평균 손실 계산 및 출력
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # 최상의 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_file)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the preprocessed data.")
    parser.add_argument("--checkpoint_file", type=str, default="model_checkpoint.pth",
                        help="Path to save the model checkpoint.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        checkpoint_file=args.checkpoint_file,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
