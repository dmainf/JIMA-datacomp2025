"""
モデルをキャッシュして再利用するためのモジュール
"""

_model_cache = {}

def get_model(model_path='./data/models/sentence-transformer'):
    """
    モデルをキャッシュから取得（初回のみロード）
    M4 MacのGPU (MPS) を使用

    Parameters:
    -----------
    model_path : str
        モデルのパス

    Returns:
    --------
    SentenceTransformer
        キャッシュされたモデル
    """
    global _model_cache

    if model_path not in _model_cache:
        print(f"Loading model from {model_path}...")
        from sentence_transformers import SentenceTransformer
        import torch

        # M4 MacのGPUを使用
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {device}")

        _model_cache[model_path] = SentenceTransformer(model_path, device=device)
        print("Model loaded and cached!")
    else:
        print("Using cached model")

    return _model_cache[model_path]

def clear_cache():
    """キャッシュをクリア"""
    global _model_cache
    _model_cache = {}
    print("Model cache cleared")
