import argparse
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def preprocess_data(output_dir):
    # Kaggle 데이터 다운로드
    import kagglehub
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")

    image_dir = path + '/images'
    csv_path = path + '/styles.csv'

    # 데이터 로드
    data = pd.read_csv(csv_path, on_bad_lines='skip', usecols=['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage', 'productDisplayName'])
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

    print(f"Preprocessing completed. Data saved to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for training.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed data.")
    args = parser.parse_args()
    preprocess_data(args.output_dir)
