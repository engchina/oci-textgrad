#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル: Vision MathVista

TextGradを使用した視覚的数学問題解決を紹介します。

要件:
- OpenAI API キーが必要です。環境変数 OPENAI_API_KEY として設定してください。
"""

import io
import os
from PIL import Image
import textgrad as tg
from textgrad import get_engine
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss

def load_sample_image():
    """サンプル画像を読み込む（実際の実装では適切な画像ファイルを使用）"""
    # デモ用の小さな画像を作成
    # 実際のアプリケーションでは、MathVistaデータセットの画像を使用
    image = Image.new('RGB', (200, 100), color='white')

    # 画像をバイトデータに変換
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr

def main():
    """メイン実行関数"""
    print("TextGrad Vision MathVista チュートリアルを開始します...")

    # 環境変数の確認
    print("\n=== 環境設定の確認 ===")
    oci_compartment_id = os.getenv('OCI_COMPARTMENT_ID')
    if not oci_compartment_id:
        print("警告: OCI_COMPARTMENT_ID環境変数が設定されていません。")
        print("実際の使用時には、OCI Compartment IDを設定してください。")
    else:
        print("OCI Compartment IDが設定されています。")

    # バックワードエンジンの設定
    print("\n=== エンジンの設定 ===")
    try:
        tg.set_backward_engine("meta.llama-4-scout-17b-16e-instruct", override=True)
        print("OCI Generative AI (meta.llama-4-scout-17b-16e-instruct) バックワードエンジンが設定されました。")
    except Exception as e:
        print(f"エンジンの設定に失敗しました: {e}")
        print("デモモードで続行します。")

    # 画像の読み込み
    print("\n=== 画像の読み込み ===")
    try:
        image_data = load_sample_image()
        print("サンプル画像を読み込みました。")
        print(f"画像データサイズ: {len(image_data)} バイト")
    except Exception as e:
        print(f"画像の読み込みに失敗しました: {e}")
        return

    # 画像変数の作成
    print("\n=== 画像変数の作成 ===")
    image_variable = tg.Variable(image_data,
                                role_description="数学問題の図表",
                                requires_grad=False)
    print("画像変数が作成されました。")

    # 数学問題の定義
    print("\n=== 数学問題の定義 ===")
    math_question = """この図表を見て、以下の質問に答えてください：
1. 図表に表示されている数値は何ですか？
2. グラフの傾向を説明してください。
3. 最大値と最小値を特定してください。"""

    question_variable = tg.Variable(math_question,
                                   role_description="数学問題",
                                   requires_grad=False)
    print(f"数学問題: {math_question}")

    # 初期回答の作成
    print("\n=== 初期回答の作成 ===")
    initial_answer = """図表を分析した結果：
1. 数値は0から100の範囲にあります
2. 上昇傾向が見られます
3. 最大値は約90、最小値は約10です"""

    answer_variable = tg.Variable(initial_answer,
                                 role_description="数学問題への回答",
                                 requires_grad=True)
    print(f"初期回答: {initial_answer}")

    # マルチモーダルLLM呼び出しの設定
    print("\n=== マルチモーダルLLM呼び出しの設定 ===")
    try:
        multimodal_llm = MultimodalLLMCall("meta.llama-4-scout-17b-16e-instruct")
        print("マルチモーダルLLMが設定されました。")

        # 画像と質問を組み合わせた呼び出し
        print("\n=== 画像分析の実行 ===")
        print("実際のアプリケーションでは、ここでマルチモーダルLLMが画像を分析します。")

        # デモ用の模擬応答
        mock_response = tg.Variable(
            "この画像は数学的なグラフまたは図表を含んでいるようです。詳細な分析には実際の画像データが必要です。",
            role_description="LLMからの応答"
        )
        print(f"LLM応答: {mock_response.value}")

    except Exception as e:
        print(f"マルチモーダルLLM呼び出しに失敗しました: {e}")
        print("デモモードで続行します。")

    # 画像QA損失の使用
    print("\n=== 画像QA損失の使用 ===")
    try:
        evaluation_instruction = "この数学問題に対する回答が完全で正確かどうかを評価してください。数学的な正確性を重視して厳しく批評してください。"
        qa_loss = ImageQALoss(
            evaluation_instruction=evaluation_instruction,
            engine="meta.llama-4-scout-17b-16e-instruct"
        )
        print("ImageQALoss損失関数が作成されました。")

        # 損失の計算（実際の実装では画像、質問、回答を使用）
        print("実際のアプリケーションでは、ここで以下を実行します：")
        print("1. qa_loss(image=image_variable, question=question_variable, response=answer_variable)")
        print("2. 回答の品質を評価")
        print("3. 改善のためのフィードバックを生成")

    except Exception as e:
        print(f"ImageQALoss の作成に失敗しました: {e}")
        print("デモモードで続行します。")

    # 最適化プロセス
    print("\n=== 最適化プロセス ===")
    print("TextGradを使用した視覚的数学問題解決の最適化プロセス：")
    print("1. 初期回答の生成")
    print("2. 画像と質問に基づく回答の評価")
    print("3. 損失の計算")
    print("4. 勾配の計算（逆伝播）")
    print("5. 回答の改善")
    print("6. 必要に応じて反復")

    # オプティマイザーの設定
    print("\n=== オプティマイザーの設定 ===")
    try:
        optimizer = tg.TGD(parameters=[answer_variable])
        print("テキスト勾配降下オプティマイザーが設定されました。")
    except Exception as e:
        print(f"オプティマイザーの設定に失敗しました: {e}")

    # MathVistaデータセットでの応用
    print("\n=== MathVistaデータセットでの応用 ===")
    print("MathVistaデータセットを使用する場合の手順：")
    print("1. データセットから画像と質問をロード")
    print("2. 各問題に対して初期回答を生成")
    print("3. TextGradを使用して回答を最適化")
    print("4. 正解率の向上を測定")
    print("5. 異なる問題タイプでの性能を評価")

    # 視覚的推論の改善
    print("\n=== 視覚的推論の改善 ===")
    print("TextGradによる視覚的推論の改善点：")
    print("- 図表の詳細な分析")
    print("- 数値の正確な読み取り")
    print("- グラフの傾向の理解")
    print("- 幾何学的関係の認識")
    print("- 数学的概念の適用")

    # 結果の評価
    print("\n=== 結果の評価 ===")
    print("最適化後の回答の評価基準：")
    print("- 数値の正確性")
    print("- 推論の論理性")
    print("- 説明の明確性")
    print("- 数学的概念の正しい適用")

    print("\nチュートリアル完了！")
    print("実際の使用時には、MathVistaデータセットと適切な画像データを使用してください。")

if __name__ == "__main__":
    main()
