#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextGrad チュートリアル: 実験的エンジン

TextGradの実験的エンジンを使用する方法を説明します。
このチュートリアルでは、OCI Generative AI を使用します。

要件:
- OCI Generative AI が設定されている必要があります
- OCI設定を行い、OCI_COMPARTMENT_ID環境変数を設定してください
"""

import textgrad
import os
import httpx
from dotenv import load_dotenv

def main():
    """メイン実行関数"""
    # 環境変数の読み込み
    load_dotenv()
    
    print("TextGrad 実験的エンジンチュートリアルを開始します...")
    
    # OCI Generative AI の設定確認
    print("\n=== OCI Generative AI の設定確認 ===")
    oci_compartment = os.getenv('OCI_COMPARTMENT_ID')
    if not oci_compartment:
        print("警告: OCI_COMPARTMENT_ID環境変数が設定されていません。")
        print("実際の使用時には、OCI Generative AI の設定を完了してください。")
    else:
        print("OCI Generative AI の設定が確認されました。")
    
    # 基本的なテキスト生成
    print("\n=== 基本的なテキスト生成 ===")
    try:
        # OCI Generative AI エンジンを使用
        engine = textgrad.get_engine("xai.grok-3")
        
        # 簡単な質問
        response = engine.generate("こんにちは、3+4はいくつですか？")
        print("エンジンからの応答:")
        print(response)
        
    except Exception as e:
        print(f"エンジンの初期化または生成に失敗しました: {e}")
        print("デモモードで続行します。")
        
        # デモ用の模擬応答
        print("模擬応答: こんにちは！3+4は7です。")
    
    # マルチモーダル機能のテスト
    print("\n=== マルチモーダル機能のテスト ===")
    try:
        # 画像データの取得
        image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
        print(f"画像を取得中: {image_url}")
        
        image_data = httpx.get(image_url).content
        print(f"画像データサイズ: {len(image_data)} バイト")
        
        # マルチモーダルエンジンを使用
        multimodal_engine = textgrad.get_engine("meta.llama-4-scout-17b-16e-instruct")
        
        # 画像とテキストを組み合わせた質問
        from textgrad.autograd import MultimodalLLMCall
        multimodal_call = MultimodalLLMCall("meta.llama-4-scout-17b-16e-instruct")
        
        # 変数の作成
        image_variable = textgrad.Variable(image_data, 
                                         role_description="分析する画像", 
                                         requires_grad=False)
        question_variable = textgrad.Variable("これは何ですか？", 
                                            role_description="画像に関する質問", 
                                            requires_grad=False)
        
        # マルチモーダル推論の実行
        response = multimodal_call([image_variable, question_variable])
        print("マルチモーダル応答:")
        print(response.value)
        
    except Exception as e:
        print(f"マルチモーダル機能のテストに失敗しました: {e}")
        print("デモモードで続行します。")
        print("模擬応答: この画像には昆虫（アリ）が写っています。")
    
    # 実験的エンジンの特徴
    print("\n=== 実験的エンジンの特徴 ===")
    print("実験的エンジンの主な特徴：")
    print("1. 新しいモデルアーキテクチャのサポート")
    print("2. 高度なキャッシュ機能")
    print("3. カスタマイズ可能な生成パラメータ")
    print("4. マルチモーダル対応")
    print("5. 実験的な最適化手法")
    
    # キャッシュ機能のデモ
    print("\n=== キャッシュ機能のデモ ===")
    print("実際のアプリケーションでは、以下のようにキャッシュを有効化できます：")
    print("engine = textgrad.get_engine('xai.grok-3', cache=True)")
    print("これにより、同じ入力に対する応答がキャッシュされ、性能が向上します。")
    
    # 高度な設定
    print("\n=== 高度な設定 ===")
    print("実験的エンジンでは、以下のような高度な設定が可能です：")
    print("- 温度パラメータの調整")
    print("- トークン数の制限")
    print("- 応答フォーマットの指定")
    print("- カスタムシステムプロンプト")
    
    # パフォーマンス最適化
    print("\n=== パフォーマンス最適化 ===")
    print("実験的エンジンのパフォーマンス最適化のヒント：")
    print("1. 適切なバッチサイズの設定")
    print("2. キャッシュの有効活用")
    print("3. 並列処理の利用")
    print("4. メモリ使用量の監視")
    print("5. ネットワーク最適化")
    
    # エラーハンドリング
    print("\n=== エラーハンドリング ===")
    print("実験的エンジンを使用する際の注意点：")
    print("- 適切な例外処理の実装")
    print("- レート制限への対応")
    print("- ネットワークエラーの処理")
    print("- モデルの可用性確認")
    
    # 今後の展望
    print("\n=== 今後の展望 ===")
    print("実験的エンジンの今後の発展：")
    print("- より多くのモデルのサポート")
    print("- 性能の向上")
    print("- 新機能の追加")
    print("- 安定性の改善")
    print("- コミュニティからのフィードバック統合")
    
    print("\nチュートリアル完了！")
    print("実験的エンジンを使用して、最新のAI機能を探索してください。")

if __name__ == "__main__":
    main()
