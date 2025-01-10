"""
PlantVillageデータセットをLlama 3.2 Vision-Instruct用の指示データセットに変換するスクリプト
"""

import os
import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定数の定義
DATA_ROOT = Path("data/processed")
OUTPUT_ROOT = Path("data/instruction_dataset")
SPLITS = ["train", "val", "test"]

# 指示テンプレート
INSTRUCTION_TEMPLATE = "この植物の画像を分析し、種類と健康状態を診断してください。必要に応じて対策も提案してください。"


def load_class_mapping() -> Dict[str, int]:
    """クラスマッピングの読み込み"""
    with open(DATA_ROOT / "class_mapping.json", "r", encoding="utf-8") as f:
        return {v: k for k, v in json.load(f).items()}  # インデックスをキーに変更


def create_output_description(class_name: str) -> str:
    """クラス名から出力の説明文を生成"""
    # クラス名をパーツに分解（例：Apple___Apple_scab -> ["Apple", "Apple_scab"]）
    parts = class_name.split("___")
    plant = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ")

    if condition.lower() == "healthy":
        return f"この画像は{plant}の健康な状態を示しています。特に対策は必要ありません。"
    else:
        return f"この画像は{plant}の{condition}による病気を示しています。早急な対策が必要です。"


def create_instruction_dataset(split: str, class_mapping: Dict[int, str]) -> List[Dict]:
    """指示データセットの作成"""
    logger.info(f"Processing {split} dataset...")
    dataset = []
    split_dir = DATA_ROOT / split

    # 各クラスディレクトリを処理
    for class_idx in tqdm(os.listdir(split_dir)):
        if not class_idx.isdigit():
            continue

        class_dir = split_dir / class_idx
        if not class_dir.is_dir():
            continue

        class_name = class_mapping[int(class_idx)]

        # 各画像を処理
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            relative_img_path = f"data/processed/{split}/{class_idx}/{img_file}"

            # データ項目の作成
            item = {
                "image_path": relative_img_path,
                "instruction": INSTRUCTION_TEMPLATE,
                "input": "",  # 追加の入力は不要
                "output": create_output_description(class_name)
            }
            dataset.append(item)

    return dataset


def save_instruction_dataset(dataset: List[Dict], split: str):
    """指示データセットの保存"""
    output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{split}.jsonl"
    logger.info(f"Saving {split} dataset to {output_file}")

    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(dataset)


def main():
    """メイン処理"""
    logger.info("Starting instruction dataset preparation...")

    # 出力ディレクトリの作成
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # クラスマッピングの読み込み
    class_mapping = load_class_mapping()

    # 各分割データセットの処理
    for split in SPLITS:
        dataset = create_instruction_dataset(split, class_mapping)
        save_instruction_dataset(dataset, split)
        logger.info(f"Processed {len(dataset)} examples for {split}")

    logger.info("Instruction dataset preparation completed!")


if __name__ == "__main__":
    main()
