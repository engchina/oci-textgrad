#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル: マルチモーダル最適化

TextGradを使用したマルチモーダル最適化を紹介します。

要件:
- OCI Generative AI の設定が必要です。環境変数 OCI_COMPARTMENT_ID として設定してください。
"""

import io
from PIL import Image
import httpx
import textgrad as tg
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss

def main():
    """メイン実行関数"""
    print("TextGrad マルチモーダル最適化チュートリアルを開始します...")

    # バックワードエンジンの設定
    print("\n=== エンジンの設定 ===")
    tg.set_backward_engine("meta.llama-4-scout-17b-16e-instruct")
    print("OCI Generative AI (meta.llama-4-scout-17b-16e-instruct) バックワードエンジンが設定されました。")

    # 画像の取得
    print("\n=== 画像の取得 ===")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    print(f"画像URL: {image_url}")

    try:
        image_data = httpx.get(image_url).content
        print("画像データを取得しました。")
    except Exception as e:
        print(f"画像の取得に失敗しました: {e}")
        return

    # 画像を Variable として定義
    print("\n=== 画像 Variable の作成 ===")
    print("TextGradでは、画像データもVariableオブジェクトに変換する必要があります。")
    image_variable = tg.Variable(image_data,
                                role_description="質問に答えるための画像",
                                requires_grad=False)
    print("画像Variableが作成されました。")

    # 質問の作成
    print("\n=== 質問の作成 ===")
    question_variable = tg.Variable("この画像に何が写っていますか？",
                                   role_description="質問",
                                   requires_grad=False)
    print(f"質問: {question_variable.value}")

    # マルチモーダルLLM呼び出し
    print("\n=== マルチモーダルLLM呼び出し ===")
    print("マルチモーダルLLMを使用して画像について質問します...")

    try:
        response = MultimodalLLMCall("meta.llama-4-scout-17b-16e-instruct")([image_variable, question_variable])
        print("回答:")
        print(response.value)
    except Exception as e:
        print(f"マルチモーダルLLM呼び出しに失敗しました: {e}")
        return

    # 画像QA損失の使用例
    print("\n=== 画像QA損失の使用例 ===")
    print("ImageQALossを使用して画像に関する質問応答を最適化できます。")

    # 最適化可能な回答を作成
    optimizable_answer = tg.Variable("これは昆虫です。",
                                    role_description="最適化可能な回答",
                                    requires_grad=True)

    # 損失関数の作成
    try:
        evaluation_instruction = "この画像に関する回答が完全で良い回答かどうかを評価してください。厳しく批評してください。"
        loss_fn = ImageQALoss(
            evaluation_instruction=evaluation_instruction,
            engine="meta.llama-4-scout-17b-16e-instruct"
        )
        print("ImageQALoss損失関数が作成されました。")

        print("最適化可能な回答:")
        print(optimizable_answer.value)

        # 損失の計算（実際の最適化は省略）
        print("\n=== 損失の計算 ===")
        print("実際のアプリケーションでは、ここで以下を実行します：")
        print("1. loss_fn(image=image_variable, question=question_variable, response=optimizable_answer)")
        print("2. 逆伝播と最適化ステップを実行して回答を改善")

    except Exception as e:
        print(f"ImageQALoss の作成に失敗しました: {e}")
        print("デモモードで続行します。")
        print("実際のアプリケーションでは、以下を実行します：")
        print("1. 画像、質問、回答を損失関数に入力")
        print("2. 回答の品質を評価")
        print("3. 改善のためのフィードバックを生成")

    # マルチモーダル最適化の応用例
    print("\n=== マルチモーダル最適化の応用例 ===")
    print("マルチモーダル最適化は以下のような用途に使用できます：")
    print("1. 画像キャプション生成の改善")
    print("2. 視覚的質問応答システムの最適化")
    print("3. 画像とテキストの対応関係の学習")
    print("4. マルチモーダル検索システムの改善")

    # 画像の表示（オプション）
    print("\n=== 画像の表示 ===")
    try:
        # 画像をPILで開いて情報を表示
        image = Image.open(io.BytesIO(image_data))
        print(f"画像サイズ: {image.size}")
        print(f"画像モード: {image.mode}")
        print("注意: 実際の画像を表示するには、適切な画像表示ライブラリを使用してください。")
    except Exception as e:
        print(f"画像の処理に失敗しました: {e}")

    print("\nチュートリアル完了！")

if __name__ == "__main__":
    main()
