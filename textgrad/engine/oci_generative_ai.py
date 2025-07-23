# OCI imports are done lazily to avoid slow startup times
oci = None
GenerativeAiInferenceClient = None
ChatDetails = None
OnDemandServingMode = None
GenericChatRequest = None
UserMessage = None
SystemMessage = None
TextContent = None
ImageContent = None
ImageUrl = None

def _import_oci():
    """Lazy import of OCI modules"""
    global oci, GenerativeAiInferenceClient, ChatDetails, OnDemandServingMode
    global GenericChatRequest, UserMessage, SystemMessage, TextContent
    global ImageContent, ImageUrl

    if oci is None:
        try:
            import oci as _oci
            from oci.generative_ai_inference import GenerativeAiInferenceClient as _GenerativeAiInferenceClient
            from oci.generative_ai_inference.models import (
                ChatDetails as _ChatDetails,
                OnDemandServingMode as _OnDemandServingMode,
                GenericChatRequest as _GenericChatRequest,
                UserMessage as _UserMessage,
                SystemMessage as _SystemMessage,
                TextContent as _TextContent,
                ImageContent as _ImageContent,
                ImageUrl as _ImageUrl
            )

            oci = _oci
            GenerativeAiInferenceClient = _GenerativeAiInferenceClient
            ChatDetails = _ChatDetails
            OnDemandServingMode = _OnDemandServingMode
            GenericChatRequest = _GenericChatRequest
            UserMessage = _UserMessage
            SystemMessage = _SystemMessage
            TextContent = _TextContent
            ImageContent = _ImageContent
            ImageUrl = _ImageUrl

        except ImportError:
            raise ImportError(
                "OCIのGenerative AIモデルを使用するには、'pip install oci'でociパッケージをインストールし、"
                "適切なOCI認証設定を行ってください。"
            )

import os
import json
import base64
from typing import List, Union
import platformdirs
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# .envファイルから環境変数を読み込み
load_dotenv()

from .base import CachedEngine, EngineLM
from .engine_utils import get_image_type_from_bytes


class BaseOCIEngine(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "あなたは役に立つ、創造的で賢いアシスタントです。"

    def __init__(
        self,
        cache_path: str,
        system_prompt: str,
        model_string: str,
        compartment_id: str,
        region: str = "us-chicago-1",
        is_multimodal: bool = False,
    ):
        super().__init__(cache_path=cache_path)
        self.system_prompt = system_prompt
        self.model_string = model_string
        self.compartment_id = compartment_id
        self.region = region
        self.is_multimodal = is_multimodal

        # OCI設定を初期化
        self._init_oci_client()

    def _init_oci_client(self):
        """OCI Generative AI Inferenceクライアントを初期化"""
        try:
            # Lazy import OCI modules
            _import_oci()

            # デフォルトのOCI設定を使用
            config = oci.config.from_file()
            config["region"] = self.region

            self.client = GenerativeAiInferenceClient(config)
        except Exception as e:
            raise ValueError(f"OCI設定の初期化に失敗しました: {e}")

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(
        self,
        content: Union[str, List[Union[str, bytes]]],
        system_prompt: str = None,
        **kwargs,
    ):
        if isinstance(content, str):
            return self._generate_from_single_prompt(
                content, system_prompt=system_prompt, **kwargs
            )
        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if has_multimodal_input and not self.is_multimodal:
                raise NotImplementedError(
                    "マルチモーダル生成はマルチモーダル対応モデルでのみサポートされています。"
                )
            return self._generate_from_multiple_input(
                content, system_prompt=system_prompt, **kwargs
            )

    def _generate_from_single_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        top_p: float = 0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_key = sys_prompt_arg + prompt
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        # メッセージを構築
        messages = []
        if sys_prompt_arg:
            messages.append(SystemMessage(content=[TextContent(text=sys_prompt_arg)]))
        messages.append(UserMessage(content=[TextContent(text=prompt)]))

        # チャットリクエストを作成
        chat_request = GenericChatRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            is_stream=False
        )

        # サービングモードを設定
        serving_mode = OnDemandServingMode(model_id=self.model_string)

        # チャット詳細を作成
        chat_details = ChatDetails(
            serving_mode=serving_mode,
            chat_request=chat_request,
            compartment_id=self.compartment_id
        )

        try:
            # APIを呼び出し
            response = self.client.chat(chat_details)

            # レスポンスからテキストを抽出
            if response.data and response.data.chat_response and response.data.chat_response.choices:
                response_text = response.data.chat_response.choices[0].message.content[0].text
                self._save_cache(cache_key, response_text)
                return response_text
            else:
                raise ValueError("APIレスポンスが期待される形式ではありません")

        except Exception as e:
            raise RuntimeError(f"OCI Generative AI API呼び出しエラー: {e}")

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content_for_oci(self, content: List[Union[str, bytes]]) -> List:
        """文字列とバイトのリストをOCI APIに渡すためのコンテンツリストに変換するヘルパー関数"""
        # Lazy import OCI modules
        _import_oci()

        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                # バイトは画像として扱われる
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode("utf-8")

                # OCI APIの画像コンテンツ形式
                image_url = ImageUrl(url=f"data:image/{image_type};base64,{base64_image}")
                image_content = ImageContent(image_url=image_url)
                formatted_content.append(image_content)

            elif isinstance(item, str):
                # テキストコンテンツ
                text_content = TextContent(text=item)
                formatted_content.append(text_content)
            else:
                raise ValueError(f"サポートされていない入力タイプ: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt=None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        top_p: float = 0.99,
    ):
        # Lazy import OCI modules
        _import_oci()

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content_for_oci(content)

        # キャッシュキーの生成（簡略化）
        cache_key = sys_prompt_arg + str(len(content)) + str(hash(str(content)))
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        # メッセージを構築
        messages = []
        if sys_prompt_arg:
            messages.append(SystemMessage(content=[TextContent(text=sys_prompt_arg)]))

        # マルチモーダルコンテンツでユーザーメッセージを作成
        messages.append(UserMessage(content=formatted_content))

        # チャットリクエストを作成
        chat_request = GenericChatRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            is_stream=False
        )

        # サービングモードを設定
        serving_mode = OnDemandServingMode(model_id=self.model_string)

        # チャット詳細を作成
        chat_details = ChatDetails(
            serving_mode=serving_mode,
            chat_request=chat_request,
            compartment_id=self.compartment_id
        )

        try:
            # APIを呼び出し
            response = self.client.chat(chat_details)

            # レスポンスからテキストを抽出
            if response.data and response.data.chat_response and response.data.chat_response.choices:
                response_text = response.data.chat_response.choices[0].message.content[0].text
                self._save_cache(cache_key, response_text)
                return response_text
            else:
                raise ValueError("APIレスポンスが期待される形式ではありません")

        except Exception as e:
            raise RuntimeError(f"OCI Generative AI API呼び出しエラー: {e}")


class ChatOCI(BaseOCIEngine):
    def __init__(
        self,
        model_string: str = "xai.grok-3",
        system_prompt: str = BaseOCIEngine.DEFAULT_SYSTEM_PROMPT,
        compartment_id: str = None,
        region: str = "us-chicago-1",
        is_multimodal: bool = False,
        **kwargs,
    ):
        """
        OCI Generative AIエンジンの初期化

        :param model_string: 使用するモデル名（例: "xai.grok-3"）
        :param system_prompt: システムプロンプト
        :param compartment_id: OCIコンパートメントID
        :param region: OCIリージョン
        :param is_multimodal: マルチモーダルモデルかどうか
        """
        if compartment_id is None:
            # .envファイルまたは環境変数からOCI_COMPARTMENT_OCIDを取得
            compartment_id = os.getenv("OCI_COMPARTMENT_OCID") or os.getenv("OCI_COMPARTMENT_ID")
            if compartment_id is None:
                raise ValueError(
                    "OCI_COMPARTMENT_OCID環境変数を.envファイルに設定するか、compartment_idパラメータを指定してください。"
                )

        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_oci_{model_string}.db")

        super().__init__(
            cache_path=cache_path,
            system_prompt=system_prompt,
            model_string=model_string,
            compartment_id=compartment_id,
            region=region,
            is_multimodal=is_multimodal
        )
