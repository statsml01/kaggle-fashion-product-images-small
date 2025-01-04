import argparse
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from PIL import Image
import numpy as np


def text_pipeline(product_name, vocab, tokenizer):
    return torch.tensor([vocab[token] for token in tokenizer(product_name)], dtype=torch.int64)

# 이미지 데이터 전처리 함수


def preprocess_image(img_path, target_size=(128, 128)):
    """
    이미지 전처리 함수: 리사이즈, 정규화, PyTorch 텐서 변환
    """
    try:
        img = Image.open(img_path).convert('RGB')  # 이미지 로드
        img = img.resize(target_size, Image.BILINEAR)  # 크기 조정
        img_array = np.array(img) / 255.0  # 정규화 [0, 1]
        img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)  # 텐서 변환
        return img_tensor
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)


def preprocess_data(output_dir):
    # Kaggle 데이터 다운로드
    import kagglehub
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")

    image_dir = path + '/images'
    csv_path = path + '/styles.csv'

    # 데이터 로드
    data = pd.read_csv(csv_path, on_bad_lines='skip', usecols=[
                       'id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage', 'productDisplayName'])
    data['image_path'] = data['id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    data = data.dropna()

    # 타겟 라벨 인코딩
    label_encoders = {}
    for col in ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # 텍스트 토큰화 및 Vocabulary 생성
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for product_name in data_iter['productDisplayName']:
            yield tokenizer(product_name)

    vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<pad>"])
    vocab.set_default_index(vocab["<pad>"])  # OOV 처리

    # 데이터 분리 및 저장
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    torch.save(vocab, os.path.join(output_dir, "vocab.pth"))
    torch.save(label_encoders, os.path.join(output_dir, "label_encoders.pth"))
    torch.save(num_classes_per_label, os.path.join(output_dir, "num_classes_per_label.pth"))

    print(f"Preprocessing completed. Data saved to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for training.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed data.")
    args = parser.parse_args()
    preprocess_data(args.output_dir)
