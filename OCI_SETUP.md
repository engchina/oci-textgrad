# Oracle OCI Generative AI セットアップガイド

このプロジェクトは、Oracle Cloud Infrastructure (OCI) Generative AI サービスの xai.grok-3 モデルをサポートするように修正されています。

## 前提条件

1. Oracle Cloud Infrastructure アカウント
2. OCI Generative AI サービスへのアクセス権限
3. Python 3.11 以上

## インストール

### 1. Conda 仮想環境の作成と依存関係のインストール

```bash
# Python 3.11 で仮想環境を作成
conda create -n oci-textgrad python=3.11 -y

# 環境をアクティベート
conda activate oci-textgrad

# 依存関係をインストール
pip install -r requirements.txt
pip install -e .
```

### 2. OCI SDK の設定

OCI Python SDK を使用するため、認証設定が必要です。

#### 方法1: 設定ファイルを使用（推奨）

1. OCI CLI をインストール:
```bash
pip install oci-cli
```

2. OCI CLI を設定:
```bash
oci setup config
```

これにより `~/.oci/config` ファイルが作成されます。

#### 方法2: 環境変数を使用

以下の環境変数を設定:

```bash
export OCI_CONFIG_FILE="~/.oci/config"
export OCI_CONFIG_PROFILE="DEFAULT"
```

### 3. コンパートメント ID の設定

#### 方法1: .env ファイルを使用（推奨）

プロジェクトルートに `.env` ファイルを作成し、以下を記述:

```bash
OCI_COMPARTMENT_OCID=ocid1.compartment.oc1..your_compartment_id
```

#### 方法2: 環境変数として設定

```bash
export OCI_COMPARTMENT_OCID="ocid1.compartment.oc1..your_compartment_id"
# または従来の名前でも可
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your_compartment_id"
```

## 使用方法

### 基本的な使用例

```python
from textgrad.engine import get_engine

# .envファイルに OCI_COMPARTMENT_OCID が設定されている場合、
# 自動的に読み込まれます

# エンジンを取得
engine = get_engine("grok-3")  # または "xai.grok-3"

# テキスト生成
response = engine.generate("こんにちは、元気ですか？")
print(response)
```

### マルチモーダル使用例（Llama-4 Scout）

```python
from textgrad.engine import get_engine

# マルチモーダル対応モデルを取得
engine = get_engine("llama-4-scout")  # または "meta.llama-4-scout-17b-16e-instruct"

# 画像とテキストの組み合わせ
with open("image.jpg", "rb") as f:
    image_data = f.read()

response = engine.generate([
    "この画像について説明してください。",
    image_data
])
print(response)
```

### 直接エンジンを使用

```python
from textgrad.engine.oci_generative_ai import ChatOCI

engine = ChatOCI(
    model_string="xai.grok-3",
    compartment_id="your_compartment_id",
    region="us-chicago-1",  # 適切なリージョンを指定
    system_prompt="あなたは役に立つアシスタントです。"
)

response = engine.generate("日本の首都はどこですか？")
print(response)
```

### TextGrad での使用

```python
from textgrad import Variable, get_engine
from textgrad.autograd import LLMCall

# エンジンを設定
engine = get_engine("grok-3")

# LLM呼び出しを作成
llm_call = LLMCall(engine)

# 変数を作成
prompt = Variable("日本の歴史について教えてください", role_description="ユーザーの質問")

# 推論を実行
response = llm_call(prompt)
print(response.value)
```

## 利用可能なモデル

現在サポートされているモデル:

- `xai.grok-3` - xAI の Grok-3 モデル（マルチモーダル対応）

ショートカット名:
- `grok-3` → `xai.grok-3`
- `grok` → `xai.grok-3`

## 設定オプション

### ChatOCI クラスのパラメータ

- `model_string`: 使用するモデル名（デフォルト: "xai.grok-3"）
- `compartment_id`: OCI コンパートメント ID
- `region`: OCI リージョン（デフォルト: "us-chicago-1"）
- `system_prompt`: システムプロンプト
- `is_multimodal`: マルチモーダル対応フラグ（デフォルト: False）

### 利用可能なリージョン

- `us-chicago-1`
- `us-ashburn-1`
- その他の OCI Generative AI 対応リージョン

## トラブルシューティング

### よくあるエラー

1. **"OCI_COMPARTMENT_ID環境変数が設定されていません"**
   - 環境変数 `OCI_COMPARTMENT_ID` を設定してください

2. **"OCI設定の初期化に失敗しました"**
   - `~/.oci/config` ファイルが正しく設定されているか確認
   - OCI CLI で `oci iam user get --user-id <your-user-id>` を実行して認証をテスト

3. **"APIレスポンスが期待される形式ではありません"**
   - モデル名が正しいか確認
   - リージョンでそのモデルが利用可能か確認

### デバッグ

テストスクリプトを実行して設定を確認:

```bash
python test_oci_engine.py
```

## 制限事項

1. 現在、マルチモーダル入力（画像など）の完全なサポートは実装中です
2. ストリーミング応答は現在サポートされていません
3. 一部の高度な機能（function calling など）は今後のバージョンで追加予定

## サポート

問題が発生した場合は、以下を確認してください:

1. OCI アカウントの権限設定
2. Generative AI サービスの利用可能性
3. 適切なリージョンの選択
4. コンパートメント ID の正確性

## 変更履歴

- OpenAI サポートを削除
- OCI Generative AI サポートを追加
- xai.grok-3 モデルのサポート
- 日本語対応の改善
