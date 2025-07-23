#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル: 実験的エンジンを使用したプリミティブ

実験的エンジンを使用してTextGradの基本的なプリミティブを学習します。

要件:
- OCI Generative AI が設定されている必要があります
- OCI設定を行い、OCI_COMPARTMENT_ID環境変数を設定してください
"""

from textgrad.engine import get_engine
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent
from textgrad.loss import TextLoss
from dotenv import load_dotenv
import os

def main():
    """メイン実行関数"""
    # 環境変数の読み込み
    load_dotenv()
    
    print("TextGrad 実験的エンジンプリミティブチュートリアルを開始します...")
    
    # OCI Generative AI の設定確認
    print("\n=== OCI Generative AI の設定確認 ===")
    oci_compartment = os.getenv('OCI_COMPARTMENT_ID')
    if not oci_compartment:
        print("警告: OCI_COMPARTMENT_ID環境変数が設定されていません。")
        print("実際の使用時には、OCI Generative AI の設定を完了してください。")
    else:
        print("OCI Generative AI の設定が確認されました。")
    
    # Variable の紹介
    print("\n=== Variable の紹介 ===")
    print("Variables は TextGrad における PyTorch のテンソルに相当します。")
    print("Variables は勾配を追跡し、データを管理します。")
    print("Variables には以下の引数が必要です：")
    print("1. data: 変数が保持するデータ")
    print("2. role_description: 計算グラフにおける変数の役割の説明")
    print("3. requires_grad: (オプション) 勾配が必要かどうかのブール値")
    
    # タイポのある文を作成
    x = Variable("A sntence with a typo", role_description="入力文", requires_grad=True)
    print(f"\n初期の変数値: {x.value}")
    print(f"初期の勾配: {x.gradients}")
    print(f"役割の説明: {x.role_description}")
    print(f"勾配が必要: {x.requires_grad}")
    
    # Engine の紹介
    print("\n=== Engine の紹介 ===")
    print("Engine は LLM との相互作用を抽象化したものです。")
    print("実験的エンジンでは、より高度な機能が利用できます。")
    
    try:
        # 実験的エンジンの初期化（OCI Generative AI を使用）
        engine = get_engine("xai.grok-3")
        print("OCI Generative AI (xai.grok-3) 実験的エンジンが初期化されました。")
        
        # エンジンのテスト
        response = engine.generate("こんにちは、調子はどうですか？")
        print(f"エンジンのレスポンス: {response}")
        
    except Exception as e:
        print(f"エンジンの初期化に失敗しました: {e}")
        print("デモモードで続行します。")
        
        # デモ用の模擬エンジン
        class MockEngine:
            def generate(self, prompt):
                return "こんにちは！調子は良好です。ありがとうございます。"
        
        engine = MockEngine()
        response = engine.generate("こんにちは、調子はどうですか？")
        print(f"模擬エンジンのレスポンス: {response}")
    
    # Loss の紹介
    print("\n=== Loss の紹介 ===")
    print("Loss は PyTorch の損失関数に相当します。")
    print("TextLoss は文字列に対して損失を評価します。")
    print("実験的エンジンでは、より詳細な損失評価が可能です。")
    
    system_prompt = Variable("この文の正確性を評価してください", role_description="システムプロンプト")
    
    try:
        loss = TextLoss(system_prompt, engine=engine)
        print("損失関数が作成されました。")
    except Exception as e:
        print(f"損失関数の作成に失敗しました: {e}")
        print("デモモードで続行します。")
        
        # デモ用の模擬損失関数
        class MockLoss:
            def __call__(self, variable):
                return Variable("この文にはタイポがあります", role_description="損失評価")
        
        loss = MockLoss()
        print("模擬損失関数が作成されました。")
    
    # Optimizer の紹介
    print("\n=== Optimizer の紹介 ===")
    print("Optimizer は PyTorch のオプティマイザーに相当します。")
    print("requires_grad=True の変数を更新します。")
    print("注意: これはテキストオプティマイザーです！すべての操作がテキストで行われます！")
    print("実験的エンジンでは、より高度な最適化戦略が利用できます。")
    
    try:
        optimizer = TextualGradientDescent(parameters=[x], engine=engine)
        print("実験的エンジンを使用したオプティマイザーが作成されました。")
    except Exception as e:
        print(f"オプティマイザーの作成に失敗しました: {e}")
        print("デモモードで続行します。")
        
        # デモ用の模擬オプティマイザー
        class MockOptimizer:
            def step(self):
                x.set_value("A sentence without a typo")
            def zero_grad(self):
                pass
        
        optimizer = MockOptimizer()
        print("模擬オプティマイザーが作成されました。")
    
    # すべてを組み合わせる
    print("\n=== すべてを組み合わせる ===")
    print("実験的エンジンを使用した最適化ステップを実行します...")
    
    try:
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
        
    except Exception as e:
        print(f"最適化プロセスでエラーが発生しました: {e}")
        print("デモモードで模擬最適化を実行します。")
        
        # デモ用の最適化
        print("模擬損失値: この文にはタイポがあります")
        print("模擬逆伝播が完了しました。")
        optimizer.step()
        print("模擬最適化ステップが完了しました。")
        print(f"最適化後の変数値: {x.value}")
    
    # 勾配のリセット
    print("\n=== 勾配のリセット ===")
    print("複数の最適化ステップを実行する場合は、各ステップ後に勾配をリセットする必要があります。")
    optimizer.zero_grad()
    print("勾配がリセットされました。")
    
    # 実験的エンジンの高度な機能
    print("\n=== 実験的エンジンの高度な機能 ===")
    print("実験的エンジンでは、以下の高度な機能が利用できます：")
    print("1. 詳細なキャッシュ機能")
    print("2. カスタマイズ可能な生成パラメータ")
    print("3. 高度なエラーハンドリング")
    print("4. パフォーマンス監視")
    print("5. 実験的な最適化アルゴリズム")
    
    # デバッグとモニタリング
    print("\n=== デバッグとモニタリング ===")
    print("実験的エンジンでは、以下のデバッグ機能が利用できます：")
    print("- 詳細なログ出力")
    print("- 中間結果の保存")
    print("- パフォーマンス指標の追跡")
    print("- エラーの詳細分析")
    
    # 最適化のヒント
    print("\n=== 最適化のヒント ===")
    print("実験的エンジンを効果的に使用するためのヒント：")
    print("1. 適切なバッチサイズの設定")
    print("2. キャッシュの有効活用")
    print("3. 実験的パラメータの調整")
    print("4. メモリ使用量の監視")
    print("5. 定期的な性能評価")
    
    print("\nチュートリアル完了！")
    print("実験的エンジンを使用して、より高度なTextGrad機能を探索してください。")

if __name__ == "__main__":
    main()
