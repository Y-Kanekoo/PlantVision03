"""
チェックポイント管理ユーティリティ
学習の途中経過を保存し、必要に応じて復元できるようにします
"""

import os
import json
import logging
from pathlib import Path
import torch
from typing import Dict, Optional

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/checkpoint.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """チェックポイントの管理クラス"""

    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        """
        初期化

        Args:
            checkpoint_dir: チェックポイントを保存するディレクトリ
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """メタデータの読み込み"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "last_checkpoint": None,
                "checkpoints": []
            }

    def _save_metadata(self):
        """メタデータの保存"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        step: int,
        loss: float,
        metrics: Dict
    ):
        """
        チェックポイントの保存

        Args:
            model: 保存するモデル
            optimizer: オプティマイザの状態
            scheduler: スケジューラの状態（オプション）
            epoch: 現在のエポック
            step: 現在のステップ
            loss: 現在の損失値
            metrics: 評価指標
        """
        checkpoint_path = self.checkpoint_dir / \
            f"checkpoint_epoch_{epoch}_step_{step}.pt"

        # チェックポイントの保存
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "metrics": metrics
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        # メタデータの更新
        checkpoint_info = {
            "path": str(checkpoint_path),
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "metrics": metrics
        }

        self.metadata["checkpoints"].append(checkpoint_info)
        self.metadata["last_checkpoint"] = checkpoint_info
        self._save_metadata()

        logger.info(f"チェックポイントを保存しました: {checkpoint_path}")

    def load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Optional[Dict]:
        """
        最新のチェックポイントを読み込む

        Args:
            model: 読み込み先のモデル
            optimizer: 読み込み先のオプティマイザ
            scheduler: 読み込み先のスケジューラ（オプション）

        Returns:
            チェックポイントの情報（存在しない場合はNone）
        """
        if not self.metadata["last_checkpoint"]:
            logger.info("利用可能なチェックポイントがありません")
            return None

        checkpoint_path = self.metadata["last_checkpoint"]["path"]

        try:
            checkpoint = torch.load(checkpoint_path)

            # モデルの状態を復元
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            logger.info(f"チェックポイントを読み込みました: {checkpoint_path}")
            logger.info(
                f"エポック: {checkpoint['epoch']}, ステップ: {checkpoint['step']}")
            logger.info(f"損失値: {checkpoint['loss']:.4f}")

            return checkpoint

        except Exception as e:
            logger.error(f"チェックポイントの読み込みに失敗しました: {str(e)}")
            return None

    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """
        古いチェックポイントを削除

        Args:
            keep_last_n: 保持するチェックポイントの数
        """
        if len(self.metadata["checkpoints"]) <= keep_last_n:
            return

        checkpoints_to_remove = self.metadata["checkpoints"][:-keep_last_n]
        self.metadata["checkpoints"] = self.metadata["checkpoints"][-keep_last_n:]

        for checkpoint in checkpoints_to_remove:
            try:
                os.remove(checkpoint["path"])
                logger.info(f"古いチェックポイントを削除しました: {checkpoint['path']}")
            except Exception as e:
                logger.error(f"チェックポイントの削除に失敗しました: {str(e)}")

        self._save_metadata()
