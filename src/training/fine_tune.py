"""
Llama 3.2 Vision-Instructモデルのファインチューニングスクリプト
LoRAを使用して効率的な学習を実現します
"""

import os
import json
import math
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from PIL import Image
import evaluate
from tqdm import tqdm
import transformers
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    MllamaProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import wandb
from einops import rearrange
import torchvision.transforms as T
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.text import BLEUScore

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """モデルの引数"""
    base_model_path: str
    output_dir: str
    torch_dtype: str = "float16"
    max_tiles: int = 4
    image_aspect_ratio: str = "auto"


@dataclass
class DataArguments:
    """データの引数"""
    train_file: str
    validation_file: str
    test_file: str
    max_source_length: int = 256
    max_target_length: int = 256
    image_size: int = 224
    image_mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225])
    max_num_images: int = 1
    max_num_tiles: int = 4


def load_config() -> Dict:
    """設定ファイルの読み込み"""
    config_path = Path("configs/fine_tuning_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_aspect_ratio(image: Image.Image) -> Tuple[int, int]:
    """画像のアスペクト比を計算"""
    width, height = image.size
    gcd = math.gcd(width, height)
    return width // gcd, height // gcd


def split_image_into_tiles(
    image: Image.Image,
    max_tiles: int,
    target_size: Tuple[int, int]
) -> List[Image.Image]:
    """画像をタイルに分割"""
    width, height = image.size
    aspect_ratio = get_aspect_ratio(image)

    if aspect_ratio[0] > aspect_ratio[1]:
        # 横長の画像
        num_tiles = min(max_tiles, aspect_ratio[0] // aspect_ratio[1])
        tile_width = width // num_tiles
        tiles = [
            image.crop((i * tile_width, 0, (i + 1) * tile_width, height))
            for i in range(num_tiles)
        ]
    else:
        # 縦長の画像
        num_tiles = min(max_tiles, aspect_ratio[1] // aspect_ratio[0])
        tile_height = height // num_tiles
        tiles = [
            image.crop((0, i * tile_height, width, (i + 1) * tile_height))
            for i in range(num_tiles)
        ]

    # タイルをリサイズ
    tiles = [tile.resize(target_size) for tile in tiles]
    return tiles


def preprocess_image(
    image_path: str,
    data_args: DataArguments
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """画像の前処理"""
    image = Image.open(image_path).convert('RGB')

    # タイルに分割
    tiles = split_image_into_tiles(
        image,
        data_args.max_num_tiles,
        (data_args.image_size, data_args.image_size)
    )

    # 前処理の定義
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=data_args.image_mean, std=data_args.image_std)
    ])

    # タイルの処理
    processed_tiles = []
    for tile in tiles:
        processed_tiles.append(transform(tile))

    # パディング
    num_tiles = len(processed_tiles)
    while len(processed_tiles) < data_args.max_num_tiles:
        processed_tiles.append(torch.zeros_like(processed_tiles[0]))

    # タイルの結合
    tiles_tensor = torch.stack(processed_tiles)

    # マスクの作成
    mask = torch.zeros(data_args.max_num_tiles, dtype=torch.bool)
    mask[:num_tiles] = True

    return tiles_tensor, mask, num_tiles


def preprocess_function(examples: Dict, processor, data_args: DataArguments) -> Dict:
    """データの前処理"""
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "pixel_values": [],
        "pixel_mask": []
    }

    for instruction, image_path, output in zip(
        examples["instruction"],
        examples["image_path"],
        examples["output"]
    ):
        # 画像の処理
        pixel_values, pixel_mask, num_tiles = preprocess_image(
            image_path,
            data_args
        )

        # テキストの処理
        text_inputs = processor(
            text=instruction,
            add_special_tokens=True,
            max_length=data_args.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        text_outputs = processor(
            text=output,
            add_special_tokens=True,
            max_length=data_args.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        model_inputs["input_ids"].append(text_inputs.input_ids)
        model_inputs["attention_mask"].append(text_inputs.attention_mask)
        model_inputs["labels"].append(text_outputs.input_ids)
        model_inputs["pixel_values"].append(pixel_values)
        model_inputs["pixel_mask"].append(pixel_mask)

    # バッチ化
    for key in model_inputs:
        if key in ["pixel_values", "pixel_mask"]:
            model_inputs[key] = torch.stack(model_inputs[key])
        else:
            model_inputs[key] = torch.cat(model_inputs[key])

    return model_inputs


def create_model_and_processor(model_args: ModelArguments) -> tuple:
    """モデルとプロセッサーの作成"""
    logger.info("Loading model and processor...")

    # プロセッサーの読み込み
    processor = AutoProcessor.from_pretrained(
        model_args.base_model_path,
        trust_remote_code=True
    )

    # モデルの読み込み
    model = MllamaForConditionalGeneration.from_pretrained(
        model_args.base_model_path,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        trust_remote_code=True
    )

    # 4bit量子化モデルの準備
    model = prepare_model_for_kbit_training(model)

    return model, processor


def create_lora_model(model: nn.Module, config: Dict) -> nn.Module:
    """LoRAモデルの作成"""
    logger.info("Creating LoRA model...")

    lora_config = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["alpha"],
        target_modules=config["lora_config"]["target_modules"],
        lora_dropout=config["lora_config"]["dropout"],
        bias=config["lora_config"]["bias"],
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def compute_metrics(eval_preds):
    """評価指標の計算"""
    predictions, labels = eval_preds

    # BLEUスコアの計算
    bleu = BLEUScore()
    bleu_score = bleu(predictions, labels)

    # F1スコアの計算（マルチクラス）
    f1 = MulticlassF1Score(num_classes=38)  # クラス数
    f1_score = f1(predictions, labels)

    return {
        "bleu": bleu_score,
        "f1": f1_score
    }


def create_trainer(
    model: nn.Module,
    processor,
    config: Dict,
    train_dataset,
    eval_dataset,
    data_collator
) -> Trainer:
    """Trainerの作成"""
    logger.info("Creating trainer...")

    training_args = TrainingArguments(
        output_dir=config["model_config"]["output_dir"],
        num_train_epochs=config["training_config"]["num_epochs"],
        per_device_train_batch_size=config["training_config"]["batch_size"],
        gradient_accumulation_steps=config["training_config"]["gradient_accumulation_steps"],
        learning_rate=config["training_config"]["learning_rate"],
        weight_decay=config["training_config"]["weight_decay"],
        warmup_ratio=config["training_config"]["warmup_ratio"],
        evaluation_strategy=config["training_config"]["evaluation_strategy"],
        eval_steps=config["training_config"]["eval_steps"],
        save_strategy=config["training_config"]["save_strategy"],
        save_steps=config["training_config"]["save_steps"],
        logging_steps=config["training_config"]["logging_steps"],
        max_grad_norm=config["training_config"]["max_grad_norm"],
        lr_scheduler_type=config["training_config"]["lr_scheduler_type"],
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    return trainer


def main():
    """メイン処理"""
    logger.info("Starting fine-tuning process...")

    # 設定の読み込み
    config = load_config()

    # wandbの初期化
    wandb.init(project="plant-vision", name="llama-3.2-vision-finetune")

    # データ引数の作成
    data_args = DataArguments(
        train_file=config["data_config"]["train_file"],
        validation_file=config["data_config"]["validation_file"],
        test_file=config["data_config"]["test_file"],
        max_source_length=config["data_config"]["max_source_length"],
        max_target_length=config["data_config"]["max_target_length"],
        image_size=config["data_config"]["image_size"],
        image_mean=config["data_config"]["image_mean"],
        image_std=config["data_config"]["image_std"],
        max_num_images=config["data_config"]["max_num_images"],
        max_num_tiles=config["data_config"]["max_num_tiles"]
    )

    # モデル引数の作成
    model_args = ModelArguments(
        base_model_path=config["model_config"]["base_model_path"],
        output_dir=config["model_config"]["output_dir"],
        torch_dtype=config["model_config"]["torch_dtype"],
        max_tiles=config["model_config"]["max_tiles"],
        image_aspect_ratio=config["model_config"]["image_aspect_ratio"]
    )

    # モデルとプロセッサーの作成
    model, processor = create_model_and_processor(model_args)

    # LoRAモデルの作成
    model = create_lora_model(model, config)

    # データセットの読み込み
    logger.info("Loading datasets...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_args.train_file,
            "validation": data_args.validation_file
        }
    )

    # データの前処理
    logger.info("Preprocessing datasets...")
    train_dataset = dataset["train"].map(
        lambda x: preprocess_function(x, processor, data_args),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )
    eval_dataset = dataset["validation"].map(
        lambda x: preprocess_function(x, processor, data_args),
        batched=True,
        remove_columns=dataset["validation"].column_names,
        num_proc=4
    )

    # データコレーターの作成
    data_collator = DataCollatorForSeq2Seq(
        processor,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # トレーナーの作成
    trainer = create_trainer(
        model,
        processor,
        config,
        train_dataset,
        eval_dataset,
        data_collator
    )

    try:
        # 学習の実行
        logger.info("Starting training...")
        train_result = trainer.train()

        # 学習結果の保存
        logger.info("Saving final model...")
        trainer.save_model()

        # 学習メトリクスの保存
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # 評価の実行
        logger.info("Running evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        # wandbのクリーンアップ
        wandb.finish()


if __name__ == "__main__":
    main()
