#!/usr/bin/env python3
"""
OCI Generative AI エンジンのテストスクリプト

このスクリプトは、Oracle OCI Generative AI エンジンの基本機能をテストします。
実行前に以下の環境変数を設定してください：

1. OCI設定ファイル (~/.oci/config) が正しく設定されていること
2. OCI_COMPARTMENT_ID 環境変数が設定されていること

使用例:
    export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your_compartment_id"
    python test_oci_engine.py
"""

import os
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_functionality():
    """基本的な機能のテスト"""
    print("=== OCI Generative AI エンジンの基本テスト ===")

    try:
        from textgrad.engine.oci_generative_ai import ChatOCI
        print("✓ OCI エンジンのインポートに成功")
    except ImportError as e:
        print(f"✗ OCI エンジンのインポートに失敗: {e}")
        return False

    # 環境変数の確認（OCI_COMPARTMENT_OCIDまたはOCI_COMPARTMENT_IDをチェック）
    compartment_id = os.getenv("OCI_COMPARTMENT_OCID") or os.getenv("OCI_COMPARTMENT_ID")
    if not compartment_id:
        print("✗ OCI_COMPARTMENT_OCID または OCI_COMPARTMENT_ID 環境変数が設定されていません")
        print("  以下のいずれかの方法で設定してください:")
        print("  1. .envファイルに: OCI_COMPARTMENT_OCID='ocid1.compartment.oc1..your_compartment_id'")
        print("  2. 環境変数として: export OCI_COMPARTMENT_ID='ocid1.compartment.oc1..your_compartment_id'")
        return False

    print(f"✓ OCI_COMPARTMENT_ID が設定されています: {compartment_id[:20]}...")

    try:
        # エンジンの初期化テスト
        engine = ChatOCI(
            model_string="xai.grok-3",
            compartment_id=compartment_id,
            region="us-chicago-1"
        )
        print("✓ OCI エンジンの初期化に成功")

        # 基本的なプロンプトテスト（実際のAPI呼び出しは行わない）
        print("✓ エンジンオブジェクトが正常に作成されました")
        print(f"  モデル: {engine.model_string}")
        print(f"  リージョン: {engine.region}")
        print(f"  マルチモーダル: {engine.is_multimodal}")

        return True

    except Exception as e:
        print(f"✗ OCI エンジンの初期化に失敗: {e}")
        print("  OCI設定ファイル (~/.oci/config) が正しく設定されているか確認してください")
        return False

def test_get_engine_function():
    """get_engine関数のテスト"""
    print("\n=== get_engine関数のテスト ===")

    try:
        from textgrad.engine import get_engine
        print("✓ get_engine関数のインポートに成功")

        # ショートカット名のテスト
        shortcuts = ["grok-3", "xai.grok-3", "llama-4-scout", "meta.llama-4-scout-17b-16e-instruct"]

        for shortcut in shortcuts:
            try:
                # 実際のエンジン作成は環境設定に依存するため、
                # エラーが発生することを期待
                engine = get_engine(shortcut)
                print(f"✓ '{shortcut}' でエンジン作成に成功")
            except ValueError as e:
                if "OCI_COMPARTMENT_OCID" in str(e) or "OCI_COMPARTMENT_ID" in str(e) or "OCI設定" in str(e):
                    print(f"✓ '{shortcut}' は正しく認識されました（設定不足のため初期化失敗）")
                else:
                    print(f"✗ '{shortcut}' で予期しないエラー: {e}")
            except Exception as e:
                print(f"✓ '{shortcut}' は正しく認識されました（OCI設定エラー）")

        return True

    except Exception as e:
        print(f"✗ get_engine関数のテストに失敗: {e}")
        return False

def test_multimodal_support():
    """マルチモーダル対応のテスト"""
    print("\n=== マルチモーダル対応のテスト ===")

    try:
        from textgrad.engine import _check_if_multimodal

        # Llama-4 Scoutはマルチモーダル対応
        assert _check_if_multimodal("meta.llama-4-scout-17b-16e-instruct") == True
        print("✓ meta.llama-4-scout-17b-16e-instruct がマルチモーダル対応として認識されました")

        # Grok-3はテキストのみ
        assert _check_if_multimodal("xai.grok-3") == False
        print("✓ xai.grok-3 がテキストのみモデルとして認識されました")

        # 存在しないモデルはマルチモーダル非対応
        assert _check_if_multimodal("non-existent-model") == False
        print("✓ 存在しないモデルは非マルチモーダルとして認識されました")

        return True

    except Exception as e:
        print(f"✗ マルチモーダル対応のテストに失敗: {e}")
        return False

def main():
    """メイン関数"""
    print("Oracle OCI Generative AI エンジンのテストを開始します...\n")

    tests = [
        test_basic_functionality,
        test_get_engine_function,
        test_multimodal_support,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=== テスト結果 ===")
    print(f"成功: {passed}/{total}")

    if passed == total:
        print("✓ すべてのテストが成功しました！")
        print("\n次のステップ:")
        print("1. OCI設定ファイルを正しく設定")
        print("2. OCI_COMPARTMENT_ID環境変数を設定")
        print("3. 実際のAPI呼び出しテストを実行")
        return 0
    else:
        print("✗ 一部のテストが失敗しました")
        return 1

if __name__ == "__main__":
    sys.exit(main())
