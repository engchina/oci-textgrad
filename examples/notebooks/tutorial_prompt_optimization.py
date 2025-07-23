#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル: プロンプト最適化

このチュートリアルでは、プロンプト最適化を実行します。

要件:
- OpenAI API キーが必要です。環境変数 OPENAI_API_KEY として設定してください。
"""

import argparse
import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random

def set_seed(seed):
    """シード値を設定する関数"""
    np.random.seed(seed)
    random.seed(seed)

def eval_sample(item, eval_fn, model):
    """
    プロンプト内の質問に対する回答が良い回答かどうかを評価する関数
    """
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="言語モデルへのクエリ")
    y = tg.Variable(y, requires_grad=False, role_description="クエリの正解")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)

def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    """データセット全体を評価する関数"""
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"精度: {np.mean(accuracy_list)}")
    return accuracy_list

def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    """検証を実行し、性能が悪化した場合は前のプロンプトに戻す関数"""
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("検証性能: ", val_performance)
    print("前回の性能: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if val_performance < previous_performance:
        print(f"拒否されたプロンプト: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)

def main():
    """メイン実行関数"""
    # 環境変数の読み込み
    load_dotenv(override=True)

    print("TextGrad プロンプト最適化チュートリアルを開始します...")

    # シード設定
    set_seed(12)

    # エンジンの初期化
    print("\n=== エンジンの初期化 ===")
    llm_api_eval = tg.get_engine(engine_name="xai.grok-3")
    llm_api_test = tg.get_engine(engine_name="xai.grok-3")
    tg.set_backward_engine(llm_api_eval, override=True)
    print("OCI Generative AI (xai.grok-3) エンジンが初期化されました。")

    # データとタスクの読み込み
    print("\n=== データとタスクの読み込み ===")
    train_set, val_set, test_set, eval_fn = load_task("BBH_object_counting", evaluation_api=llm_api_eval)
    print("訓練/検証/テストセットの長さ: ", len(train_set), len(val_set), len(test_set))
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()
    print("開始システムプロンプト:")
    print(STARTING_SYSTEM_PROMPT)

    # データローダーの作成
    train_loader = tg.tasks.DataLoader(train_set, batch_size=3, shuffle=True)

    # モデルとオプティマイザーの設定
    print("\n=== モデルとオプティマイザーの設定 ===")
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT,
                                requires_grad=True,
                                role_description="QAタスクの動作と戦略を指定する、ある程度有能な言語モデルへの構造化システムプロンプト")
    model = tg.BlackboxLLM(llm_api_test, system_prompt)
    optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

    # 結果の初期化
    results = {"test_acc": [], "prompt": [], "validation_acc": []}

    # 初期性能の評価
    print("\n=== 初期性能の評価 ===")
    results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
    results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
    results["prompt"].append(system_prompt.get_value())

    print(f"初期テスト精度: {np.mean(results['test_acc'][-1]):.4f}")
    print(f"初期検証精度: {np.mean(results['validation_acc'][-1]):.4f}")

    # 訓練ループ
    print("\n=== 訓練ループの開始 ===")
    for epoch in range(3):
        print(f"\nエポック {epoch + 1}/3")
        for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            pbar.set_description(f"訓練ステップ {steps}. エポック {epoch}")
            optimizer.zero_grad()
            losses = []

            # バッチ内の各サンプルを処理
            for (x, y) in zip(batch_x, batch_y):
                x = tg.Variable(x, requires_grad=False, role_description="言語モデルへのクエリ")
                y = tg.Variable(y, requires_grad=False, role_description="クエリの正解")
                response = model(x)
                try:
                    eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
                except:
                    eval_output_variable = eval_fn([x, y, response])
                losses.append(eval_output_variable)

            # 損失の合計と逆伝播
            total_loss = tg.sum(losses)
            total_loss.backward()
            optimizer.step()

            # 検証と復元
            run_validation_revert(system_prompt, results, model, eval_fn, val_set)

            print("現在のシステムプロンプト: ", system_prompt.value[:100] + "...")

            # テスト精度の評価
            test_acc = eval_dataset(test_set, eval_fn, model)
            results["test_acc"].append(test_acc)
            results["prompt"].append(system_prompt.get_value())

            print(f"テスト精度: {np.mean(test_acc):.4f}")

            if steps == 3:  # 各エポックで4ステップのみ実行
                break

    # 最終結果の表示
    print("\n=== 最終結果 ===")
    final_test_acc = np.mean(results["test_acc"][-1])
    final_val_acc = np.mean(results["validation_acc"][-1])
    print(f"最終テスト精度: {final_test_acc:.4f}")
    print(f"最終検証精度: {final_val_acc:.4f}")
    print(f"最終プロンプト: {system_prompt.value}")

    print("\nチュートリアル完了！")

if __name__ == "__main__":
    main()
