"""
Llama 3.2 11B Vision Instructモデルのダウンロードと量子化を行うスクリプト
"""

import os
from pathlib import Path
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定数の定義
# Llama 3.2 11B Vision Instructモデル
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MODEL_DIR = Path("models/llama_4bit")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_4bit_quantization() -> BitsAndBytesConfig:
    """4bit量子化の設定"""
    return BitsAndBytesConfig(
        load_in_4bit=True,              # 4bit量子化を有効化
        bnb_4bit_quant_type="nf4",      # 正規化浮動小数点量子化
        bnb_4bit_compute_dtype=torch.float16,  # 計算時の精度
        bnb_4bit_use_double_quant=True  # 二重量子化を使用
    )


def download_and_quantize_model():
    """モデルのダウンロードと量子化"""
    logger.info("Starting model download and quantization...")

    # 量子化設定
    quantization_config = setup_4bit_quantization()

    # トークナイザーのダウンロード
    logger.info("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(MODEL_DIR)
    logger.info("Tokenizer saved successfully")

    # モデルのダウンロードと量子化
    logger.info("Downloading and quantizing model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.save_pretrained(MODEL_DIR)
    logger.info("Model quantized and saved successfully")


def verify_cuda_availability():
    """CUDA環境の確認"""
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. GPU acceleration is required for this model.")
        return False

    logger.info(
        f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    return True


def main():
    """メイン処理"""
    # CUDA環境の確認
    if not verify_cuda_availability():
        return

    # モデルのダウンロードと量子化
    download_and_quantize_model()

    logger.info("Model preparation completed!")


if __name__ == "__main__":
    main()
