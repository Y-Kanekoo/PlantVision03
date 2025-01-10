"""
PlantVillage データセットの前処理を行うスクリプト
"""

import os
import shutil
from pathlib import Path
import json
from typing import Dict, List, Tuple
import random
import logging

import numpy as np
from PIL import Image
import albumentations as A
from tqdm import tqdm

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定数の定義
RAW_DATA_DIR = Path("data/raw/plantvillage dataset/color")
PROCESSED_DATA_DIR = Path("data/processed")
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# データ拡張の設定
train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.GaussNoise(p=1),
        A.GaussNoise(p=1),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                       rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.ElasticTransform(p=0.3),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        A.Emboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
])


def create_directories() -> None:
    """必要なディレクトリを作成"""
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def get_class_mapping() -> Dict[str, int]:
    """クラスとインデックスのマッピングを作成"""
    classes = sorted([d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # マッピングを保存
    with open(PROCESSED_DATA_DIR / "class_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)

    logger.info(f"Created class mapping with {len(classes)} classes")
    return class_to_idx


def split_data(class_to_idx: Dict[str, int]) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """データを訓練、検証、テストセットに分割"""
    all_images = []

    # すべての画像パスとラベルを収集
    for class_dir in RAW_DATA_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        class_idx = class_to_idx[class_dir.name]
        class_images = [(img_path, class_idx)
                        for img_path in class_dir.glob("*.JPG")]
        all_images.extend(class_images)

    # シャッフルして分割
    random.shuffle(all_images)
    total = len(all_images)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)

    train_data = all_images[:train_size]
    val_data = all_images[train_size:train_size + val_size]
    test_data = all_images[train_size + val_size:]

    logger.info(
        f"Split data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data


def process_and_save_images(image_data: List[Tuple[Path, int]], output_dir: Path, apply_augmentation: bool = False) -> None:
    """画像の処理と保存"""
    for img_path, class_idx in tqdm(image_data, desc=f"Processing {output_dir.name} images"):
        # 画像の読み込みと前処理
        img = Image.open(img_path)
        img = img.convert('RGB')

        # データ拡張（訓練データのみ）
        if apply_augmentation:
            img_array = np.array(img)
            augmented = train_transform(image=img_array)
            img_array = augmented['image']
            img = Image.fromarray(img_array)

        # 保存先のディレクトリを作成
        save_dir = output_dir / str(class_idx)
        save_dir.mkdir(exist_ok=True)

        # 画像を保存
        save_path = save_dir / \
            f"{img_path.stem}_{random.randint(0, 1000000)}.jpg"
        img.save(save_path, quality=95)


def main():
    """メイン処理"""
    logger.info("Starting data preprocessing...")

    # ディレクトリの作成
    create_directories()

    # クラスマッピングの作成
    class_to_idx = get_class_mapping()

    # データの分割
    train_data, val_data, test_data = split_data(class_to_idx)

    # データの処理と保存
    process_and_save_images(train_data, TRAIN_DIR, apply_augmentation=True)
    process_and_save_images(val_data, VAL_DIR)
    process_and_save_images(test_data, TEST_DIR)

    logger.info("Data preprocessing completed!")


if __name__ == "__main__":
    main()
