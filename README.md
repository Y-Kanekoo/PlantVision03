# **設計書: マルチモーダルAIを活用した植物診断システム開発**

## **プロジェクトの概要**

MLLM (マルチモーダル大規模モデル)の一つである **Llama 3.2 11B** を使用して、植物診断を行うAIシステムを構築します。主に下記の3つの目的を達成するため、ファインチューニングと量子化技術を組み合わせます。

### **目的**
1. **植物の種類の分類**  
   - データセットに基づき、植物の種類を特定します。
2. **病気・異常の診断**  
   - 画像を解析し、病気や異常を特定します。
3. **具体的な対策の提案**  
   - 診断結果をもとに、有効な対策を提案します。

---

## **プロジェクト構造**

下記の構造に基づき、ディレクトリ構造を構築します。

```
project_root/
├── data/
│   ├── raw/             # 生データセット (PlantVillage)
│   ├── processed/       # 前処理後のデータ
│   └── annotations/     # 必要なラベルデータ
├── models/
│   ├── llama_4bit/      # 量子化済みモデル
│   └── fine_tuned/      # ファインチューニング済みモデル
├── src/
│   ├── data_processing/ # データ前処理スクリプト
│   ├── training/        # モデル学習スクリプト
│   ├── inference/       # 推論用スクリプト
│   └── evaluation/      # 評価スクリプト
├── tests/               # テストコード
├── results/             # 学習や評価結果
├── configs/             # 設定ファイル (ハイパーパラメータなど)
├── requirements.txt     # 使用するPythonライブラリの一覧
├── .gitignore           # Git管理対象外ファイルの定義
├── README.md            # プロジェクトの概要と使い方
├── venv/                # 仮想環境用のディレクトリ
└── CONTRIBUTING.md      # プロジェクトへの貢献ガイド
```

---

## **使用技術と環境**

- **ハードウェア**
  - GPU: NVIDIA RTX 4070 Ti (VRAM 12GB)
  - RAM: 32GB
- **OS**: Windows 11
- **ソフトウェアスタック**
  - Python 3.10.11
  - CUDA 12.4
  - ライブラリ: 
    - `torch==2.5.1`
    - `transformers==4.47.1`
    - `accelerate==1.2.1`
    - `bitsandbytes==0.45.0`
    - `wandb==0.19.1`

---

## **仮想環境のセットアップ手順**

1. **仮想環境の作成**
   ```powershell
   python -m venv venv
   ```
2. **仮想環境の有効化**
   - **Windows**: 
     ```powershell
     .\venv\Scripts\activate
     ```
   - **Linux/Mac**:
     ```bash
     source venv/bin/activate
     ```
3. **必要なライブラリのインストール**
   ```powershell
   pip install -r requirements.txt
   ```
4. **仮想環境の終了**
   ```powershell
   deactivate
   ```

---

## **データセット**

- **使用データセット**: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **データ内容**:
  - 植物画像 (種類、健康状態のラベル付き)
  - 約3万枚以上の画像を含む。
  - 健康な植物と病気の植物がバランスよく分布。

---

## **評価指標**

1. **分類タスク**
   - **精度 (Accuracy)**
   - **F1スコア (F1 Score)**
2. **診断タスク**
   - **ROC-AUC**
   - **平均精度 (mAP)**
3. **提案タスク**
   - ユーザー評価 (主観的に提案の有用性を評価)

---

## **開発フロー**

### **ステップ 1: 環境構築**
- PowerShellでディレクトリと空ファイルを作成。
- 仮想環境を構築し、必要なライブラリをインストール。

### **ステップ 2: データセットの準備**
- PlantVillage Datasetを`data/raw/`に配置。
- `src/data_processing/`にデータ前処理スクリプトを実装。

### **ステップ 3: モデルの準備**
- HuggingFaceから事前学習済みモデルをダウンロード。
- `bitsandbytes`で4-bit量子化を実施。

### **ステップ 4: ファインチューニング**
- `src/training/`に学習スクリプトを作成。
- LoRAを用いた効率的なファインチューニング。

### **ステップ 5: 推論**
- `src/inference/`に推論スクリプトを実装。
- サンプル画像でテストを実行。

### **ステップ 6: 評価**
- `src/evaluation/`に評価スクリプトを作成。
- 精度、F1スコア、ROC-AUCを測定。

### **ステップ 7: デプロイ**
- 簡易Webサーバーを構築し、ローカルで動作確認。

---

## **不明点・質問事項**

1. **診断結果の具体性**: 提案内容の詳細度をどのレベルまで求めるか。
2. **評価基準の目標値**: 評価指標に対する具体的な目標値を設定する必要があるか。
3. **デプロイの環境**: ローカルのみで完結するのか、クラウド環境でのデプロイも考慮するのか。

---

