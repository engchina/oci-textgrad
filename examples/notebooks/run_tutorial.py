#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル実行スクリプト

このスクリプトを使用して、利用可能なチュートリアルを選択して実行できます。
"""

import os
import sys
import subprocess
from pathlib import Path

# 利用可能なチュートリアル
TUTORIALS = {
    "1": {
        "name": "プリミティブ",
        "file": "tutorial_primitives.py",
        "description": "TextGradの基本的なプリミティブ（Variable、Engine、Loss、Optimizer）を学習"
    },
    "2": {
        "name": "プロンプト最適化",
        "file": "tutorial_prompt_optimization.py",
        "description": "プロンプト最適化の実行方法を学習"
    },
    "3": {
        "name": "ソリューション最適化",
        "file": "tutorial_solution_optimization.py",
        "description": "数学問題の解を最適化する方法を学習"
    },
    "4": {
        "name": "マルチモーダル最適化",
        "file": "tutorial_multimodal.py",
        "description": "画像とテキストを組み合わせた最適化を学習"
    },
    "5": {
        "name": "テスト時損失（コード用）",
        "file": "tutorial_test_time_loss_code.py",
        "description": "コードの最適化のためのテスト時損失を学習"
    },
    "6": {
        "name": "Vision MathVista",
        "file": "tutorial_vision_mathvista.py",
        "description": "視覚的数学問題解決を学習"
    },
    "7": {
        "name": "OCI Generative AI",
        "file": "tutorial_oci_generative_ai.py",
        "description": "OCI Generative AI を使用したテキストモデルの使用方法を学習"
    }
}

def check_requirements():
    """必要な要件をチェック"""
    print("=== 要件チェック ===")

    # Python バージョンチェック
    python_version = sys.version_info
    print(f"Python バージョン: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 7):
        print("警告: Python 3.7以上が推奨されます")

    # 必要なライブラリのチェック
    required_packages = [
        "textgrad",
        "dotenv",
        "PIL",
        "httpx"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == "dotenv":
                import dotenv
            elif package == "PIL":
                import PIL
            elif package == "httpx":
                import httpx
            elif package == "textgrad":
                import textgrad
            print(f"✓ {package} が利用可能")
        except ImportError:
            print(f"✗ {package} が見つかりません")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n以下のパッケージをインストールしてください:")
        for package in missing_packages:
            if package == "dotenv":
                print(f"pip install python-dotenv")
            elif package == "PIL":
                print(f"pip install pillow")
            else:
                print(f"pip install {package}")
        return False

    # 環境変数のチェック
    print("\n=== 環境変数チェック ===")
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✓ OPENAI_API_KEY が設定されています")
    else:
        print("✗ OPENAI_API_KEY が設定されていません")
        print("  多くのチュートリアルでOpenAI APIキーが必要です")

    oci_compartment = os.getenv('OCI_COMPARTMENT_OCID')
    if oci_compartment:
        print("✓ OCI_COMPARTMENT_OCID が設定されています")
    else:
        print("- OCI_COMPARTMENT_OCID が設定されていません（一部のチュートリアルで必要）")

    return True

def display_menu():
    """メニューを表示"""
    print("\n" + "="*60)
    print("TextGrad チュートリアル選択メニュー")
    print("="*60)

    for key, tutorial in TUTORIALS.items():
        print(f"{key}. {tutorial['name']}")
        print(f"   {tutorial['description']}")
        print()

    print("0. 終了")
    print("r. 要件チェック")
    print("a. すべてのチュートリアルを順番に実行")
    print("="*60)

def run_tutorial(tutorial_file):
    """指定されたチュートリアルを実行"""
    script_dir = Path(__file__).parent
    tutorial_path = script_dir / tutorial_file

    if not tutorial_path.exists():
        print(f"エラー: {tutorial_file} が見つかりません")
        return False

    print(f"\n{tutorial_file} を実行中...")
    print("-" * 50)

    try:
        # Pythonスクリプトを実行
        result = subprocess.run([sys.executable, str(tutorial_path)],
                              capture_output=False,
                              text=True)

        if result.returncode == 0:
            print("-" * 50)
            print(f"✓ {tutorial_file} が正常に完了しました")
            return True
        else:
            print("-" * 50)
            print(f"✗ {tutorial_file} の実行中にエラーが発生しました")
            return False

    except Exception as e:
        print(f"エラー: {e}")
        return False

def run_all_tutorials():
    """すべてのチュートリアルを順番に実行"""
    print("\nすべてのチュートリアルを順番に実行します...")

    success_count = 0
    total_count = len(TUTORIALS)

    for key in sorted(TUTORIALS.keys()):
        tutorial = TUTORIALS[key]
        print(f"\n{'='*60}")
        print(f"チュートリアル {key}: {tutorial['name']}")
        print(f"{'='*60}")

        if run_tutorial(tutorial['file']):
            success_count += 1

        # 次のチュートリアルに進む前に確認
        if key != str(total_count):  # 最後のチュートリアルでない場合
            response = input("\n次のチュートリアルに進みますか？ (y/n/q): ").lower()
            if response == 'q':
                break
            elif response == 'n':
                continue

    print(f"\n実行完了: {success_count}/{total_count} のチュートリアルが成功しました")

def main():
    """メイン関数"""
    print("TextGrad チュートリアル実行スクリプトへようこそ！")

    # 初期要件チェック
    if not check_requirements():
        print("\n要件を満たしていません。必要なパッケージをインストールしてから再実行してください。")
        return

    while True:
        display_menu()

        try:
            choice = input("選択してください: ").strip()

            if choice == "0":
                print("終了します。")
                break
            elif choice == "r":
                check_requirements()
            elif choice == "a":
                run_all_tutorials()
            elif choice in TUTORIALS:
                tutorial = TUTORIALS[choice]
                print(f"\n{tutorial['name']} を実行します...")
                run_tutorial(tutorial['file'])
            else:
                print("無効な選択です。もう一度選択してください。")

        except KeyboardInterrupt:
            print("\n\n中断されました。")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
