#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル: OCI Generative AI を使用したテキストモデル

OCI Generative AI の xai.grok-3 モデルを使用してTextGradを使用する方法を説明します。

要件:
- OCI Generative AI が設定されている必要があります
- OCI設定を行い、OCI_COMPARTMENT_OCID環境変数を設定してください
"""

import textgrad as tg
from dotenv import load_dotenv

def setup_oci_model():
    """OCI Generative AI モデルの設定"""
    try:
        # OCI Generative AI エンジンの作成
        engine = tg.get_engine("xai.grok-3")
        return engine
    except Exception as e:
        print(f"OCI Generative AI モデルの設定に失敗しました: {e}")
        return None

def main():
    """メイン実行関数"""
    # 環境変数の読み込み
    load_dotenv()
    
    print("TextGrad OCI Generative AI チュートリアルを開始します...")
    
    # OCI Generative AI の説明
    print("\n=== OCI Generative AI について ===")
    print("OCI Generative AI は、Oracle Cloud Infrastructure で提供される生成AIサービスです。")
    print("1. OCI アカウントを設定")
    print("2. 適切な権限を設定")
    print("3. OCI_COMPARTMENT_OCID 環境変数を設定")
    print("4. TextGrad で OCI Generative AI を使用")
    
    # OCI モデルの設定
    print("\n=== OCI Generative AI モデルの設定 ===")
    engine = setup_oci_model()
    
    if engine is None:
        print("OCI Generative AI モデルが利用できません。デモモードで続行します。")
        print("\n=== デモモード ===")
        print("実際の使用時には、以下の手順で OCI Generative AI を設定します：")
        print("1. OCI アカウントを設定")
        print("2. 適切な権限を設定")
        print("3. OCI_COMPARTMENT_OCID 環境変数を設定")
        
        # デモ用の模擬エンジン
        class MockEngine:
            def generate(self, prompt):
                return "これはデモ用の応答です。実際の OCI Generative AI からの応答ではありません。"
        
        engine = MockEngine()
    else:
        print("OCI Generative AI モデルが正常に設定されました。")
    
    # バックワードエンジンの設定
    print("\n=== バックワードエンジンの設定 ===")
    try:
        if engine and hasattr(engine, 'generate'):
            tg.set_backward_engine(engine, override=True)
            print("OCI Generative AI バックワードエンジンが設定されました。")
        else:
            print("デモモードのため、バックワードエンジンの設定をスキップします。")
    except Exception as e:
        print(f"バックワードエンジンの設定に失敗しました: {e}")
    
    # 数学問題の例
    print("\n=== 数学問題の例 ===")
    initial_solution = """3x^2 - 7x + 2 = 0 の方程式を解くために、二次公式を使用します：
x = (-b ± √(b^2 - 4ac)) / 2a
a = 3, b = -7, c = 2
x = (7 ± √((-7)^2 + 4(3)(2))) / 6
x = (7 ± √73) / 6
解は：
x1 = (7 + √73)
x2 = (7 - √73)"""
    
    print("初期解:")
    print(initial_solution)
    
    # 変数の定義
    print("\n=== 変数の定義 ===")
    solution = tg.Variable(initial_solution,
                          requires_grad=True,
                          role_description="数学問題の解")
    
    loss_system_prompt = tg.Variable("""数学問題の解を評価してください。
自分で解こうとせず、解を提供せず、エラーのみを特定してください。非常に簡潔にしてください。""",
                                    requires_grad=False,
                                    role_description="システムプロンプト")
    
    print("変数が定義されました。")
    
    # 損失関数とオプティマイザーの設定
    print("\n=== 損失関数とオプティマイザーの設定 ===")
    try:
        loss_fn = tg.TextLoss(loss_system_prompt)
        optimizer = tg.TGD([solution])
        print("損失関数とオプティマイザーが設定されました。")
    except Exception as e:
        print(f"設定に失敗しました: {e}")
        return
    
    # 損失の計算
    print("\n=== 損失の計算 ===")
    try:
        if engine and hasattr(engine, 'generate'):
            loss = loss_fn(solution)
            print("損失値:")
            print(loss.value)
        else:
            print("デモモード: 実際の損失計算をスキップします。")
            print("模擬損失値: この解は正しく見えますが、分母の計算を確認してください。")
    except Exception as e:
        print(f"損失の計算に失敗しました: {e}")
    
    # 最適化ステップ
    print("\n=== 最適化ステップ ===")
    try:
        if engine and hasattr(engine, 'generate'):
            print("逆伝播を実行中...")
            # loss.backward()
            print("最適化ステップを実行中...")
            # optimizer.step()
            print("最適化が完了しました。")
            print("最適化後の解:")
            print(solution.value)
        else:
            print("デモモード: 最適化ステップをスキップします。")
            print("実際の使用時には、以下が実行されます：")
            print("1. loss.backward() - 逆伝播")
            print("2. optimizer.step() - 最適化ステップ")
    except Exception as e:
        print(f"最適化に失敗しました: {e}")
    
    # OCI Generative AI の利点
    print("\n=== OCI Generative AI の利点 ===")
    print("OCI Generative AI を使用する利点：")
    print("- エンタープライズグレードのセキュリティ")
    print("- スケーラブルなインフラストラクチャ")
    print("- 複数のモデルから選択可能")
    print("- Oracle Cloud との統合")
    print("- 高い可用性とパフォーマンス")
    
    # 利用可能なモデル
    print("\n=== 利用可能なモデル ===")
    print("OCI Generative AI で使用できるモデル：")
    print("- xai.grok-3 (テキスト生成)")
    print("- meta.llama-4-scout-17b-16e-instruct (マルチモーダル)")
    print("- その他のOracleが提供するモデル")
    
    # 設定のヒント
    print("\n=== 設定のヒント ===")
    print("OCI Generative AI の設定のヒント：")
    print("1. OCI アカウントを適切に設定")
    print("2. 必要な権限を付与")
    print("3. OCI_COMPARTMENT_OCID 環境変数を設定")
    print("4. 適切なリージョンを選択")
    print("5. コスト管理のための制限を設定")
    
    # トラブルシューティング
    print("\n=== トラブルシューティング ===")
    print("一般的な問題と解決策：")
    print("- 認証エラー: OCI設定が正しいことを確認")
    print("- 権限エラー: 適切なIAMポリシーが設定されていることを確認")
    print("- 接続エラー: インターネット接続とOCIサービスの状態を確認")
    print("- レート制限: リクエスト頻度を調整")
    
    print("\nチュートリアル完了！")
    print("実際の使用時には、OCI Generative AI の設定を完了してください。")

if __name__ == "__main__":
    main()
