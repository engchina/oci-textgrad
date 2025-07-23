#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル: プリミティブ

TextGradの基本的なプリミティブ（Variable、Engine、Loss、Optimizer）を紹介します。

要件:
- OCI Generative AI が設定されている必要があります
- OCI設定を行い、OCI_COMPARTMENT_ID環境変数を設定してください
"""

# 必要なライブラリのインポート
from textgrad.engine import get_engine
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent
from textgrad.loss import TextLoss
from dotenv import load_dotenv

def main():
    """メイン実行関数"""
    # 環境変数の読み込み
    load_dotenv()

    print("TextGrad プリミティブチュートリアルを開始します...")

    # Variable の紹介
    print("\n=== Variable の紹介 ===")
    print("Variables は TextGrad における PyTorch のテンソルに相当します。")
    print("Variables は勾配を追跡し、データを管理します。")

    # タイポのある文を作成
    x = Variable("A sntence with a typo", role_description="入力文", requires_grad=True)
    print(f"初期の変数値: {x.value}")
    print(f"初期の勾配: {x.gradients}")

    # Engine の紹介
    print("\n=== Engine の紹介 ===")
    print("Engine は LLM との相互作用を抽象化したものです。")

    # エンジンの初期化（OCI Generative AI を使用）
    engine = get_engine("xai.grok-3")
    print("OCI Generative AI (xai.grok-3) エンジンが初期化されました。")

    # エンジンのテスト
    response = engine.generate("こんにちは、調子はどうですか？")
    print(f"エンジンのレスポンス: {response}")

    # Loss の紹介
    print("\n=== Loss の紹介 ===")
    print("Loss は PyTorch の損失関数に相当します。")
    print("TextLoss は文字列に対して損失を評価します。")

    system_prompt = Variable("この文の正確性を評価してください", role_description="システムプロンプト")
    loss = TextLoss(system_prompt, engine=engine)
    print("損失関数が作成されました。")

    # Optimizer の紹介
    print("\n=== Optimizer の紹介 ===")
    print("Optimizer は PyTorch のオプティマイザーに相当します。")
    print("requires_grad=True の変数を更新します。")
    print("注意: これはテキストオプティマイザーです！すべての操作がテキストで行われます！")

    optimizer = TextualGradientDescent(parameters=[x], engine=engine)
    print("オプティマイザーが作成されました。")

    # すべてを組み合わせる
    print("\n=== すべてを組み合わせる ===")
    print("最適化ステップを実行します...")

    # 損失の計算
    l = loss(x)
    print(f"損失値: {l.value}")

    # 逆伝播
    l.backward(engine)
    print("逆伝播が完了しました。")

    # 最適化ステップ
    optimizer.step()
    print("最適化ステップが完了しました。")

    # 結果の確認
    print(f"最適化後の変数値: {x.value}")

    # 勾配のリセット
    print("\n=== 勾配のリセット ===")
    print("複数の最適化ステップを実行する場合は、各ステップ後に勾配をリセットする必要があります。")
    optimizer.zero_grad()
    print("勾配がリセットされました。")

    print("\nチュートリアル完了！")

if __name__ == "__main__":
    main()
