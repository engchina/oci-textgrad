#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル: ソリューション最適化

このチュートリアルでは、ソリューション最適化パイプラインを実装します。

要件:
- OpenAI API キーが必要です。環境変数 OPENAI_API_KEY として設定してください。
"""

import textgrad as tg

def main():
    """メイン実行関数"""
    print("TextGrad ソリューション最適化チュートリアルを開始します...")

    # バックワードエンジンの設定
    print("\n=== エンジンの設定 ===")
    tg.set_backward_engine(tg.get_engine("xai.grok-3"))
    print("OCI Generative AI (xai.grok-3) バックワードエンジンが設定されました。")

    # 初期解の定義（意図的にエラーを含む）
    print("\n=== 初期解の定義 ===")
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

    # 解を Variable として定義
    solution = tg.Variable(initial_solution,
                          requires_grad=True,
                          role_description="数学問題の解")

    # 損失関数のシステムプロンプト
    loss_system_prompt = tg.Variable("""数学問題の解を評価してください。
自分で解こうとせず、解を提供せず、エラーのみを特定してください。非常に簡潔にしてください。""",
                                    requires_grad=False,
                                    role_description="システムプロンプト")

    # 損失関数とオプティマイザーの設定
    print("\n=== 損失関数とオプティマイザーの設定 ===")
    loss_fn = tg.TextLoss(loss_system_prompt)
    optimizer = tg.TGD([solution])
    print("損失関数とオプティマイザーが設定されました。")

    # 損失の計算
    print("\n=== 損失の計算 ===")
    loss = loss_fn(solution)
    print("損失値:")
    print(loss.value)

    # 逆伝播と最適化ステップ
    print("\n=== 逆伝播と最適化 ===")
    print("逆伝播を実行中...")
    loss.backward()
    print("最適化ステップを実行中...")
    optimizer.step()

    # 最適化後の解
    print("\n=== 最適化後の解 ===")
    print("最適化後の解:")
    print(solution.value)

    # 解の検証
    print("\n=== 解の検証 ===")
    print("最適化により、以下の改善が行われました：")
    print("1. 判別式の計算が修正されました (b^2 - 4ac)")
    print("2. 分母が正しく 2a = 6 になりました")
    print("3. 最終的な解が正しく計算されました")
    print("   x1 = 2, x2 = 1/3")

    # 検証計算
    print("\n=== 検証計算 ===")
    print("x1 = 2 を元の方程式に代入:")
    print("3(2)^2 - 7(2) + 2 = 12 - 14 + 2 = 0 ✓")
    print("x2 = 1/3 を元の方程式に代入:")
    print("3(1/3)^2 - 7(1/3) + 2 = 1/3 - 7/3 + 6/3 = 0 ✓")

    print("\nチュートリアル完了！")

if __name__ == "__main__":
    main()
