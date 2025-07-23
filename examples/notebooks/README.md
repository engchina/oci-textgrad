# TextGrad チュートリアル - Python版

このディレクトリには、Jupyter NotebookからPythonスクリプトに変換されたTextGradチュートリアルが含まれています。これらのスクリプトは直接実行でき、TextGradの様々な機能を学習するのに役立ちます。

## 利用可能なチュートリアル

### 1. プリミティブ (`tutorial_primitives.py`)
**元ファイル**: `Tutorial-Primitives.ipynb`

TextGradの基本的なプリミティブを紹介します：
- Variable（変数）
- Engine（エンジン）
- Loss（損失関数）
- Optimizer（オプティマイザー）

**実行方法**:
```bash
python tutorial_primitives.py
```

### 2. プロンプト最適化 (`tutorial_prompt_optimization.py`)
**元ファイル**: `Tutorial-Prompt-Optimization.ipynb`

プロンプト最適化の実行方法を学習します：
- データセットの読み込み
- 評価関数の設定
- 訓練ループの実行
- 検証と復元

**実行方法**:
```bash
python tutorial_prompt_optimization.py
```

### 3. ソリューション最適化 (`tutorial_solution_optimization.py`)
**元ファイル**: `Tutorial-Solution-Optimization.ipynb`

数学問題の解を最適化する方法を学習します：
- 初期解の定義
- 損失関数の設定
- 逆伝播と最適化

**実行方法**:
```bash
python tutorial_solution_optimization.py
```

### 4. マルチモーダル最適化 (`tutorial_multimodal.py`)
**元ファイル**: `Tutorial-MultiModal.ipynb`

画像とテキストを組み合わせた最適化を学習します：
- 画像データの処理
- マルチモーダルLLM呼び出し
- 画像QA損失の使用

**実行方法**:
```bash
python tutorial_multimodal.py
```

### 5. テスト時損失（コード用） (`tutorial_test_time_loss_code.py`)
**元ファイル**: `Tutorial-Test-Time-Loss-for-Code.ipynb`

コードの最適化のためのテスト時損失を学習します：
- コード評価関数の定義
- FormattedLLMCallの使用
- コードの反復的改善

**実行方法**:
```bash
python tutorial_test_time_loss_code.py
```

### 6. Vision MathVista (`tutorial_vision_mathvista.py`)
**元ファイル**: `TextGrad-Vision-MathVista.ipynb`

視覚的数学問題解決を学習します：
- 画像ベースの数学問題
- 視覚的推論の最適化
- MathVistaデータセットの使用

**実行方法**:
```bash
python tutorial_vision_mathvista.py
```

### 7. OCI Generative AI (`tutorial_oci_generative_ai.py`)
**元ファイル**: `Local-Model-With-LMStudio.ipynb`

OCI Generative AI を使用したテキストモデルの使用方法を学習します：
- OCI Generative AI の設定
- xai.grok-3 モデルの使用
- エンタープライズグレードのAI推論

**実行方法**:
```bash
python tutorial_oci_generative_ai.py
```

## 前提条件

### 必要なライブラリ
```bash
pip install textgrad
pip install python-dotenv
pip install pillow
pip install httpx
pip install oci  # OCI SDK
```

### 環境変数
すべてのチュートリアルでは、以下の環境変数が必要です：

```bash
# OCI Generative AI用（すべてのチュートリアルで必要）
export OCI_COMPARTMENT_ID="your-compartment-ocid"
```

### .envファイルの使用
プロジェクトルートに`.env`ファイルを作成することもできます：

```
OCI_COMPARTMENT_ID=your-compartment-ocid
```

## 特別な要件

### OCI Generative AI チュートリアル
`tutorial_oci_generative_ai.py`を実行する前に：

1. OCI アカウントを設定
2. 適切な権限を設定
3. OCI_COMPARTMENT_ID 環境変数を設定

### マルチモーダルチュートリアル
画像処理のために追加のライブラリが必要な場合があります：

```bash
pip install pillow
pip install httpx
```

## 実行順序の推奨

初心者の方は、以下の順序でチュートリアルを実行することをお勧めします：

1. `tutorial_primitives.py` - 基本概念の理解
2. `tutorial_solution_optimization.py` - 簡単な最適化例
3. `tutorial_prompt_optimization.py` - より複雑な最適化
4. `tutorial_multimodal.py` - マルチモーダル機能
5. `tutorial_test_time_loss_code.py` - カスタム損失関数
6. `tutorial_vision_mathvista.py` - 専門的な応用
7. `tutorial_oci_generative_ai.py` - OCI Generative AI の使用

## トラブルシューティング

### 一般的な問題

1. **OCI設定エラー**: OCI_COMPARTMENT_ID環境変数が正しく設定されていることを確認
2. **インポートエラー**: 必要なライブラリがインストールされていることを確認
3. **接続エラー**: インターネット接続とOCI Generative AIサービスの状態を確認
4. **権限エラー**: 適切なIAMポリシーが設定されていることを確認

### デバッグモード

各チュートリアルには詳細なログ出力が含まれており、問題の特定に役立ちます。

## 貢献

バグを発見した場合や改善提案がある場合は、GitHubのIssueまたはPull Requestを作成してください。

## ライセンス

これらのチュートリアルは、元のTextGradプロジェクトと同じライセンスの下で提供されています。
