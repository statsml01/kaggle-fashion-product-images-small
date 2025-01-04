import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from preprop import ProductDataset

class TextImageClassifier(nn.Module):
    def __init__(self, num_classes_per_label, use_bert=False):
        super(TextImageClassifier, self).__init__()
        
        self.use_bert = use_bert
        self.num_classes_per_label = num_classes_per_label
        
        # 텍스트 인코더
        if use_bert:
            from transformers import BertModel, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
            self.text_hidden_size = 768
        else:
            self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=128, padding_idx=0)
            self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
            self.text_hidden_size = 256 * 2
        
        # 이미지 인코더
        resnet = resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*(list(resnet.children())[:-1]), nn.Flatten())
        self.image_hidden_size = resnet.fc.in_features
        
        # 레이블별 분류기
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.text_hidden_size + self.image_hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            for num_classes in num_classes_per_label
        ])
        
    def forward(self, text_input, image_input):
        if self.use_bert:
            tokens = self.tokenizer(text_input, padding=True, truncation=True, max_length=50, return_tensors="pt")
            bert_output = self.text_encoder(**tokens)
            text_features = bert_output.pooler_output
        else:
            embedded_text = self.embedding(text_input)
            lstm_output, _ = self.lstm(embedded_text)
            text_features = lstm_output[:, -1, :]
        
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

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint

def train_model(data_dir, checkpoint_file, num_epochs=3, batch_size=32, learning_rate=1e-4):
    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    vocab = torch.load(os.path.join(data_dir, "vocab.pth"))
    label_encoders = torch.load(os.path.join(data_dir, "label_encoders.pth"))
    train_dataset = ProductDataset(train_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    target_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
    num_classes_per_label = [train_data[col].nunique() for col in target_columns]
    model = TextImageClassifier(num_labels=len(target_columns), use_bert=False).to(device)

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
    parser.add_argument("--checkpoint_file", type=str, default="model_checkpoint.pth", help="Path to save the model checkpoint.")
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
