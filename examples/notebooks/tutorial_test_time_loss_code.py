#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル: コードのテスト時損失

新しいテスト時損失を定義し、コードを最適化する方法を説明します。

要件:
- OpenAI API キーが必要です。環境変数 OPENAI_API_KEY として設定してください。
"""

import textgrad as tg
import random
import time

def test_longest_increasing_subsequence(fn):
    """最長増加部分列のテスト関数"""
    test_cases = [
        ([10, 22, 9, 33, 21, 50, 41, 60], 5),
        ([7, 2, 1, 3, 8, 4, 9, 6, 5], 4),
        ([5, 4, 3, 2, 1], 1),
        ([1, 2, 3, 4, 5], 5),
        ([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], 4),
        ([10, 9, 2, 5, 3, 7, 101, 18], 4),
        ([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15], 6),
        ([7, 7, 7, 7, 7, 7, 7], 1),
        ([20, 25, 47, 35, 56, 68, 98, 101, 212, 301, 415, 500], 11),
        ([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], 1)
    ]

    for nums, expected in test_cases:
        result = fn(nums)
        assert result == expected, f"テストケース {nums} で失敗: 期待値 {expected}, 実際の値 {result}"

    print("すべてのテストケースが成功しました！")

def generate_random_test_case(size, min_value, max_value):
    """ランダムなテストケースを生成"""
    return [random.randint(min_value, max_value) for _ in range(size)]

def run_code_safely(code_string):
    """コードを安全に実行（実際の実装では適切なサンドボックスを使用）"""
    # 注意: 実際のアプリケーションでは、コードの実行には適切なサンドボックスが必要です
    print("警告: このコードは実際には実行されません。安全のため、コード実行は無効化されています。")
    print("実際に実行する場合は、適切なサンドボックス環境を使用してください。")

    # デモ用の模擬関数を返す
    def mock_function(nums):
        # 簡単な実装（デモ用）
        if not nums:
            return 0
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    return mock_function

def main():
    """メイン実行関数"""
    print("TextGrad テスト時損失コード最適化チュートリアルを開始します...")

    # 問題の定義
    print("\n=== 問題の定義 ===")
    problem_text = """最長増加部分列 (LIS)

問題文:
整数の列が与えられたとき、厳密に増加する最長の部分列の長さを求めてください。
部分列とは、元の列から一部の要素を削除（または削除しない）して得られる列で、
残りの要素の順序は変更されません。

入力:
整数のリストで表される列

出力:
最長増加部分列の長さを表す整数"""

    print(problem_text)

    # 初期解の定義
    print("\n=== 初期解の定義 ===")
    initial_solution = """
def longest_increasing_subsequence(nums):
    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    max_length = max(dp)
    lis = []

    for i in range(n - 1, -1, -1):
        if dp[i] == max_length:
            lis.append(nums[i])
            max_length -= 1

    return len(lis[::-1])
"""

    print("初期解:")
    print(initial_solution)

    # テストケースの生成
    print("\n=== テストケースの生成 ===")
    size = 1000  # 実際のテストでは10000だが、デモ用に小さくする
    min_value = 1
    max_value = 1000
    nums = generate_random_test_case(size, min_value, max_value)
    print(f"テストケースサイズ: {size}")

    # 初期解のテスト
    print("\n=== 初期解のテスト ===")
    longest_increasing_subsequence = run_code_safely(initial_solution)

    start_time = time.time()
    lis = longest_increasing_subsequence(nums)
    end_time = time.time()

    print(f"最長増加部分列の長さ: {lis}")
    print(f"実行時間: {end_time - start_time:.5f} 秒")

    # 基本テストケースでのテスト
    test_longest_increasing_subsequence(longest_increasing_subsequence)

    # TextGradを使用したコード最適化
    print("\n=== TextGradを使用したコード最適化 ===")

    # エンジンの設定
    llm_engine = tg.get_engine("xai.grok-3")
    tg.set_backward_engine(llm_engine)
    print("OCI Generative AI (xai.grok-3) LLMエンジンが設定されました。")

    # 変数の定義
    code = tg.Variable(value=initial_solution,
                      requires_grad=True,
                      role_description="最適化するコードインスタンス")

    problem = tg.Variable(problem_text,
                         requires_grad=False,
                         role_description="コーディング問題")

    # オプティマイザーの設定
    optimizer = tg.TGD(parameters=[code])
    print("オプティマイザーが設定されました。")

    # 損失関数の定義
    print("\n=== 損失関数の定義 ===")
    loss_system_prompt = """あなたはコードスニペットを評価するスマートな言語モデルです。
問題を解いたり新しいコードスニペットを提案したりはせず、
既存のソリューションを批判的に評価し、非常に簡潔なフィードバックを提供するだけです。"""

    loss_system_prompt = tg.Variable(loss_system_prompt,
                                    requires_grad=False,
                                    role_description="損失関数へのシステムプロンプト")

    instruction = """問題とコードスニペットについて考えてください。
コードは問題を解決していますか？実行時間計算量はどうですか？"""

    format_string = "{instruction}\n問題: {{problem}}\n現在のコード: {{code}}"
    format_string = format_string.format(instruction=instruction)

    fields = {"problem": None, "code": None}
    formatted_llm_call = tg.autograd.FormattedLLMCall(engine=llm_engine,
                                                      format_string=format_string,
                                                      fields=fields,
                                                      system_prompt=loss_system_prompt)

    def loss_fn(problem: tg.Variable, code: tg.Variable) -> tg.Variable:
        inputs = {"problem": problem, "code": code}
        return formatted_llm_call(inputs=inputs,
                                 response_role_description=f"{code.get_role_description()}の評価")

    print("損失関数が定義されました。")

    # 損失の計算
    print("\n=== 損失の計算 ===")
    loss = loss_fn(problem, code)
    print("損失値:")
    print(loss.value)

    # 最適化ステップ（デモ用）
    print("\n=== 最適化ステップ ===")
    print("実際のアプリケーションでは、ここで以下の手順を実行します：")
    print("1. loss.backward() - 逆伝播")
    print("2. optimizer.step() - 最適化ステップ")
    print("3. 新しいコードの評価とテスト")
    print("4. 必要に応じて複数回の反復")

    print("\n=== 最適化の利点 ===")
    print("TextGradを使用したコード最適化により、以下が可能になります：")
    print("- アルゴリズムの効率性の改善")
    print("- コードの可読性の向上")
    print("- バグの修正")
    print("- より良いプログラミング実践の適用")

    print("\nチュートリアル完了！")

if __name__ == "__main__":
    main()
