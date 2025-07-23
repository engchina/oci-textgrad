#!/bin/bash

# OCI TextGrad 環境セットアップスクリプト

echo "=== OCI TextGrad 環境セットアップ ==="

# Conda がインストールされているかチェック
if ! command -v conda &> /dev/null; then
    echo "❌ Conda がインストールされていません。"
    echo "   Anaconda または Miniconda をインストールしてください。"
    exit 1
fi

echo "✅ Conda が見つかりました"

# Python 3.11 で仮想環境を作成
echo "🔧 Python 3.11 で oci-textgrad 環境を作成中..."
conda create -n oci-textgrad python=3.11 -y

if [ $? -ne 0 ]; then
    echo "❌ 環境の作成に失敗しました"
    exit 1
fi

echo "✅ 環境の作成が完了しました"
echo ""
echo "次のステップ:"
echo "1. 環境をアクティベート: conda activate oci-textgrad"
echo "2. 依存関係をインストール: pip install -r requirements.txt"
echo "3. .env ファイルを作成して OCI_COMPARTMENT_ID を設定"
echo "4. OCI 設定ファイル (~/.oci/config) を確認"
echo ""
echo "完了後、テストを実行: python test_oci_engine.py"
