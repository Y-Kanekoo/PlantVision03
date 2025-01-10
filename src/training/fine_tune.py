"""
Llama 3.2 Vision-Instructモデルのファインチューニングスクリプト
LoRAを使用して効率的な学習を実現します
"""

import os
import json
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
import evaluate
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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


@dataclass
class DataArguments:
    """データの引数"""
    train_file: str
    validation_file: str
    test_file: str
    max_source_length: int = 512
    max_target_length: int = 512
    image_size: int = 224
    image_mean: List[float] = None
    image_std: List[float] = None


def load_config() -> Dict:
    """設定ファイルの読み込み"""
    config_path = Path("configs/fine_tuning_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image_path: str, image_size: int, mean: List[float], std: List[float]) -> torch.Tensor:
    """画像の前処理"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image_size, image_size))
    image = torch.tensor(image).float()
    image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    # 正規化
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        image = (image / 255.0 - mean) / std

    return image


def preprocess_function(examples: Dict, tokenizer, data_args: DataArguments) -> Dict:
    """データの前処理"""
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "pixel_values": []
    }

    for instruction, image_path, output in zip(
        examples["instruction"],
        examples["image_path"],
        examples["output"]
    ):
        # テキストの処理
        source = f"{instruction}"
        target = f"{output}"

        source_ids = tokenizer(
            source,
            max_length=data_args.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_ids = tokenizer(
            target,
            max_length=data_args.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 画像の処理
        image = preprocess_image(
            image_path,
            data_args.image_size,
            data_args.image_mean,
            data_args.image_std
        )

        model_inputs["input_ids"].append(source_ids["input_ids"])
        model_inputs["attention_mask"].append(source_ids["attention_mask"])
        model_inputs["labels"].append(target_ids["input_ids"])
        model_inputs["pixel_values"].append(image)

    return model_inputs


def create_model_and_tokenizer(model_args: ModelArguments) -> tuple:
    """モデルとトークナイザーの作成"""
    logger.info("Loading model and tokenizer...")

    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.base_model_path,
        trust_remote_code=True
    )

    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model_path,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        trust_remote_code=True
    )

    # 4bit量子化モデルの準備
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


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


def create_trainer(
    model: nn.Module,
    tokenizer,
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
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
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
        image_std=config["data_config"]["image_std"]
    )

    # モデル引数の作成
    model_args = ModelArguments(
        base_model_path=config["model_config"]["base_model_path"],
        output_dir=config["model_config"]["output_dir"],
        torch_dtype=config["model_config"]["torch_dtype"]
    )

    # モデルとトークナイザーの作成
    model, tokenizer = create_model_and_tokenizer(model_args)

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
        lambda x: preprocess_function(x, tokenizer, data_args),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    eval_dataset = dataset["validation"].map(
        lambda x: preprocess_function(x, tokenizer, data_args),
        batched=True,
        remove_columns=dataset["validation"].column_names
    )

    # データコレーターの作成
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # トレーナーの作成
    trainer = create_trainer(
        model,
        tokenizer,
        config,
        train_dataset,
        eval_dataset,
        data_collator
    )

    # 学習の実行
    logger.info("Starting training...")
    trainer.train()

    # モデルの保存
    logger.info("Saving model...")
    trainer.save_model()

    logger.info("Fine-tuning completed!")


if __name__ == "__main__":
    main()
