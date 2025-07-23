from .base import EngineLM

__ENGINE_NAME_SHORTCUTS__ = {
    "grok-3": "xai.grok-3",
    "llama-4-scout": "meta.llama-4-scout-17b-16e-instruct",
}

# マルチモーダル対応エンジンのリスト
__MULTIMODAL_ENGINES__ = [
    "meta.llama-4-scout-17b-16e-instruct",  # Llama-4 Scoutはマルチモーダル対応
]

def _check_if_multimodal(engine_name: str):
    return any([name == engine_name for name in __MULTIMODAL_ENGINES__])

def validate_multimodal_engine(engine):
    if not _check_if_multimodal(engine.model_string):
        raise ValueError(
            f"The engine provided is not multimodal. Please provide a multimodal engine, one of the following: {__MULTIMODAL_ENGINES__}")

def get_engine(engine_name: str, **kwargs) -> EngineLM:
    if engine_name in __ENGINE_NAME_SHORTCUTS__:
        engine_name = __ENGINE_NAME_SHORTCUTS__[engine_name]

    # OCI Generative AIエンジンのみをサポート
    if (engine_name.startswith("xai.") or
        engine_name.startswith("meta.") or
        engine_name in ["grok-3", "llama-4-scout"]):
        from .oci_generative_ai import ChatOCI
        return ChatOCI(model_string=engine_name, is_multimodal=_check_if_multimodal(engine_name), **kwargs)
    else:
        raise ValueError(f"Engine {engine_name} not supported. このプロジェクトはOCI Generative AIのみをサポートしています。利用可能なエンジン: xai.grok-3 (grok-3), meta.llama-4-scout-17b-16e-instruct (llama-4-scout)")
