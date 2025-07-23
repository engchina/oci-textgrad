#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル環境セットアップスクリプト

このスクリプトは、TextGradチュートリアルを実行するために必要な環境をセットアップします。
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Pythonバージョンをチェック"""
    print("=== Pythonバージョンチェック ===")
    version = sys.version_info
    print(f"現在のPythonバージョン: {version.major}.{version.minor}.{version.micro}")

    if version < (3, 11):
        print("❌ Python 3.11以上が必要です")
        print("Pythonを更新してください: https://www.python.org/downloads/")
        return False
    else:
        print("✅ Pythonバージョンは要件を満たしています")
        return True

def install_package(package_name, import_name=None):
    """パッケージをインストール"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✅ {package_name} は既にインストールされています")
        return True
    except ImportError:
        print(f"📦 {package_name} をインストール中...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} のインストールが完了しました")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ {package_name} のインストールに失敗しました")
            return False

def install_required_packages():
    """必要なパッケージをインストール"""
    print("\n=== 必要なパッケージのインストール ===")

    packages = [
        ("textgrad", "textgrad"),
        ("python-dotenv", "dotenv"),
        ("pillow", "PIL"),
        ("httpx", "httpx"),
        ("openai", "openai"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy")
    ]

    success_count = 0
    for package_name, import_name in packages:
        if install_package(package_name, import_name):
            success_count += 1

    print(f"\n{success_count}/{len(packages)} パッケージが正常にインストールされました")
    return success_count == len(packages)

def create_env_file():
    """環境変数ファイルを作成"""
    print("\n=== 環境変数ファイルの作成 ===")

    env_file = Path(".env")

    if env_file.exists():
        print("✅ .envファイルは既に存在します")
        return True

    print("📝 .envファイルを作成します...")

    # ユーザーからAPIキーを取得
    print("\nOCI設定（必須 - 後で手動で設定することもできます）:")

    oci_compartment = input("OCI Compartment OCIDを入力してください（スキップする場合はEnter）: ").strip()
    openai_key = input("OpenAI APIキーを入力してください（参考用、スキップする場合はEnter）: ").strip()

    # .envファイルの内容を作成
    env_content = """# TextGrad チュートリアル用環境変数
# 以下の値を実際の値に置き換えてください

# OCI Generative AI用（すべてのチュートリアルで必要）
"""

    if oci_compartment:
        env_content += f"OCI_COMPARTMENT_OCID={oci_compartment}\n"
    else:
        env_content += "OCI_COMPARTMENT_OCID=your-oci-compartment-ocid-here\n"

    env_content += """
# OpenAI API キー（参考用、現在は使用されていません）
"""

    if openai_key:
        env_content += f"# OPENAI_API_KEY={openai_key}\n"
    else:
        env_content += "# OPENAI_API_KEY=your-openai-api-key-here\n"

    env_content += """
# その他の設定
# TEXTGRAD_CACHE_DIR=./cache
# TEXTGRAD_LOG_LEVEL=INFO
"""

    try:
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(env_content)
        print("✅ .envファイルが作成されました")

        if not oci_compartment:
            print("⚠️  OCI Compartment OCIDが設定されていません")
            print("   .envファイルを編集して実際のOCIDを設定してください")

        return True
    except Exception as e:
        print(f"❌ .envファイルの作成に失敗しました: {e}")
        return False

def check_system_requirements():
    """システム要件をチェック"""
    print("\n=== システム要件チェック ===")

    # OS情報
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"アーキテクチャ: {platform.machine()}")

    # メモリ情報（可能な場合）
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"メモリ: {memory.total // (1024**3)} GB")

        if memory.total < 4 * (1024**3):  # 4GB未満
            print("⚠️  メモリが少ない可能性があります。大きなモデルの使用時は注意してください")
    except ImportError:
        print("メモリ情報を取得できませんでした（psutilが必要）")

    return True

def create_sample_config():
    """サンプル設定ファイルを作成"""
    print("\n=== サンプル設定ファイルの作成 ===")

    config_content = """# TextGrad チュートリアル設定例
# このファイルは参考用です

[DEFAULT]
# デフォルトエンジン
default_engine = gpt-3.5-turbo

# キャッシュディレクトリ
cache_dir = ./cache

# ログレベル
log_level = INFO

[ENGINES]
# 利用可能なエンジン
openai_gpt35 = gpt-3.5-turbo
openai_gpt4 = gpt-4
openai_gpt4o = gpt-4o

[TUTORIALS]
# チュートリアル固有の設定
max_iterations = 10
batch_size = 3
"""

    config_file = Path("textgrad_config.ini")

    try:
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content)
        print("✅ サンプル設定ファイルが作成されました")
        return True
    except Exception as e:
        print(f"❌ 設定ファイルの作成に失敗しました: {e}")
        return False

def display_next_steps():
    """次のステップを表示"""
    print("\n" + "="*60)
    print("🎉 セットアップが完了しました！")
    print("="*60)

    print("\n次のステップ:")
    print("1. .envファイルを編集して実際のAPIキーを設定")
    print("2. チュートリアルを実行:")
    print("   python run_tutorial.py")
    print("\n個別のチュートリアルを実行:")
    print("   python tutorial_primitives.py")
    print("   python tutorial_prompt_optimization.py")
    print("   など...")

    print("\n📚 詳細情報:")
    print("   README.md ファイルを参照してください")

    print("\n🔧 トラブルシューティング:")
    print("   - APIキーが正しく設定されていることを確認")
    print("   - インターネット接続を確認")
    print("   - 必要に応じてファイアウォール設定を確認")

def main():
    """メイン関数"""
    print("TextGrad チュートリアル環境セットアップ")
    print("="*50)

    # Pythonバージョンチェック
    if not check_python_version():
        return

    # システム要件チェック
    check_system_requirements()

    # パッケージインストール
    if not install_required_packages():
        print("❌ 一部のパッケージのインストールに失敗しました")
        print("手動でインストールを試してください")

    # 環境変数ファイル作成
    create_env_file()

    # サンプル設定ファイル作成
    create_sample_config()

    # 次のステップを表示
    display_next_steps()

if __name__ == "__main__":
    main()
