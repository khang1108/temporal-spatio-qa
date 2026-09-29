"""
Checkpoint utilities for automatic downloading and verification of pre-trained model weights (.pt).
"""

import os
import sys
import urllib.request
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from tqdm import tqdm


# Default local checkpoint paths for each reproduced model
DEFAULT_CHECKPOINT_PATHS: Dict[str, str] = {
    "roberta": "experiments/checkpoints/roberta/best_model.pt",
    "multiqa": "experiments/checkpoints/multiqa/best_model.pt",
    "stcqa": "experiments/checkpoints/stcqa/best_model.pt",
}

# Default remote download URLs (e.g. GitHub Releases or HuggingFace)
# Can be overridden via environment variables or CLI arguments
DEFAULT_CHECKPOINT_URLS: Dict[str, str] = {
    "roberta": os.environ.get(
        "ROBERTA_CHECKPOINT_URL",
        "https://github.com/khang1108/temporal-spatio-qa/releases/download/v1.0.0/roberta_best_model.pt"
    ),
    "multiqa": os.environ.get(
        "MULTIQA_CHECKPOINT_URL",
        "https://github.com/khang1108/temporal-spatio-qa/releases/download/v1.0.0/multiqa_best_model.pt"
    ),
    "stcqa": os.environ.get(
        "STCQA_CHECKPOINT_URL",
        "https://github.com/khang1108/temporal-spatio-qa/releases/download/v1.0.0/stcqa_best_model.pt"
    ),
}


def download_file(url: str, output_path: str, chunk_size: int = 1024 * 1024):
    """
    Downloads a file from a URL to output_path with a tqdm progress bar.
    Uses custom User-Agent to ensure compatibility with GitHub / Cloudflare CDNs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = output_path + ".tmp"
    print(f"Downloading checkpoint from: {url}")
    print(f"Destination: {output_path}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) STKGQA-Downloader/1.0"}
    )
    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f:
            total_size = int(resp.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
                      desc=os.path.basename(output_path), miniters=1) as pbar:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        os.rename(temp_path, output_path)
        print(f"Successfully downloaded to {output_path}")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to download checkpoint from {url}: {e}") from e


def ensure_checkpoint(model_name: str,
                      checkpoint_path: Optional[str] = None,
                      url: Optional[str] = None,
                      force_download: bool = False) -> str:
    """
    Checks if checkpoint exists at checkpoint_path.
    If not found (or force_download=True), automatically downloads it.

    Args:
        model_name: 'roberta', 'multiqa', or 'stcqa'
        checkpoint_path: local destination path (.pt file). If None, uses default path.
        url: optional custom download URL
        force_download: if True, re-downloads even if file exists

    Returns:
        checkpoint_path
    """
    norm_name = model_name.lower().replace("-base", "")
    target_path = checkpoint_path or DEFAULT_CHECKPOINT_PATHS.get(norm_name)
    if not target_path:
        raise ValueError(f"Unknown model name '{model_name}'. Expected one of: {list(DEFAULT_CHECKPOINT_PATHS.keys())}")

    if os.path.exists(target_path) and not force_download:
        return target_path

    download_url = url or DEFAULT_CHECKPOINT_URLS.get(norm_name)
    if not download_url:
        raise ValueError(
            f"No download URL configured for model '{model_name}'. "
            f"Please specify a URL or ensure '{target_path}' exists locally."
        )

    print(f"\n[Checkpoint Notice] Checkpoint '{target_path}' not found locally.")
    print(f"Attempting automatic download for '{model_name}'...")
    try:
        download_file(download_url, target_path)
    except Exception as e:
        print(f"\n[Warning] Automatic download failed: {e}")
        print(f"If you are training from scratch, this file will be created after training.")
        print(f"Alternatively, provide a valid URL via --checkpoint_url or place the .pt file manually at {target_path}.\n")
        raise

    return target_path


def load_model_checkpoint(model: nn.Module,
                          model_name: str,
                          checkpoint_path: Optional[str] = None,
                          url: Optional[str] = None,
                          device: Optional[torch.device] = None,
                          force_download: bool = False,
                          strict: bool = True) -> nn.Module:
    """
    Logic when loading model:
    1. Checks whether the .pt file exists locally.
    2. If not found, downloads the .pt file from remote repository/URL.
    3. Loads the state_dict into the model and returns it.

    Args:
        model: PyTorch model instance
        model_name: 'roberta', 'multiqa', or 'stcqa'
        checkpoint_path: Path to the .pt file (defaults to standard checkpoint path)
        url: Optional custom download URL
        device: Target device for loading weights (e.g. 'cpu' or 'cuda')
        force_download: If True, re-downloads even if file exists locally
        strict: Whether to strictly enforce matching keys in state_dict

    Returns:
        The loaded model instance
    """
    norm_name = model_name.lower().replace("-base", "")
    target_path = checkpoint_path or DEFAULT_CHECKPOINT_PATHS.get(norm_name)
    if not target_path:
        raise ValueError(f"Unknown model name '{model_name}'. Expected one of: {list(DEFAULT_CHECKPOINT_PATHS.keys())}")

    # Check whether have .pt file, if not download .pt file
    ensure_checkpoint(
        model_name=norm_name,
        checkpoint_path=target_path,
        url=url,
        force_download=force_download
    )

    map_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from '{target_path}' to {map_device}...")
    state_dict = torch.load(target_path, map_location=map_device)
    model.load_state_dict(state_dict, strict=strict)
    print(f"Successfully loaded checkpoint into {model.__class__.__name__}.")
    return model


def load_model(model_name: str,
               checkpoint_path: Optional[str] = None,
               url: Optional[str] = None,
               device: Optional[torch.device] = None,
               **model_kwargs: Any) -> nn.Module:
    """
    Instantiates and loads one of the 3 reproduced models ('roberta', 'multiqa', 'stcqa').
    Automatically checks for the .pt checkpoint and downloads it if missing.

    Args:
        model_name: 'roberta', 'multiqa', or 'stcqa'
        checkpoint_path: Optional custom path to .pt file
        url: Optional custom download URL
        device: Device to place the model on
        **model_kwargs: Extra keyword arguments passed to the model constructor

    Returns:
        Loaded PyTorch model
    """
    norm_name = model_name.lower().replace("-base", "")
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if norm_name == "roberta":
        from baselines.roberta.model import RoBERTaBaseline
        defaults = {"model_name": "roberta-base", "num_entities": 5897, "embedding_dim": 512}
        defaults.update(model_kwargs)
        model = RoBERTaBaseline(**defaults).to(dev)

    elif norm_name == "multiqa":
        from baselines.multiqa.model import MultiQABaseline
        defaults = {"model_name": "roberta-base", "num_entities": 5897, "num_timestamps": 600, "embedding_dim": 512}
        defaults.update(model_kwargs)
        model = MultiQABaseline(**defaults).to(dev)

    elif norm_name == "stcqa":
        from src.models.stcqa.model import STCQAModel
        defaults = {
            "model_name": "roberta-base",
            "num_entities": 5897,
            "num_relations": 32,
            "num_timestamps": 170,
            "num_locations": 1352,
            "embedding_dim": 512
        }
        defaults.update(model_kwargs)
        model = STCQAModel(**defaults).to(dev)

    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: 'roberta', 'multiqa', 'stcqa'.")

    # Check whether have .pt files, if not download .pt files, then load into model
    load_model_checkpoint(
        model=model,
        model_name=norm_name,
        checkpoint_path=checkpoint_path,
        url=url,
        device=dev
    )
    return model
