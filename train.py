import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import pandas as pd
from tqdm import tqdm

from preprop import ProductDataset, TextImageClassifier

def train_model(data_dir, checkpoint_dir, num_epochs=10, batch_size=32, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    vocab = torch.load(os.path.join(data_dir, "vocab.pth"))
    train_dataset = ProductDataset(train_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모델 정의
    model = TextImageClassifier(num_labels=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_texts, batch_images, batch_targets in progress_bar:
            batch_texts = batch_texts.to(device)
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_texts, batch_images)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{running_loss/len(progress_bar):.4f}"})

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

        # 모델 저장
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth"))

    print(f"Training complete. Checkpoints saved to {checkpoint_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the preprocessed data.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save the model checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    args = parser.parse_args()
    train_model(args.data_dir, args.checkpoint_dir, args.num_epochs, args.batch_size, args.learning_rate)
