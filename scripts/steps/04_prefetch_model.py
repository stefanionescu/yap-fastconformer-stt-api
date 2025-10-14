from __future__ import annotations

import os

from huggingface_hub import snapshot_download


def main() -> None:
    repo_id = os.environ.get("HF_REPO_ID", "nvidia/parakeet-tdt-0.6b-v3")
    cache_dir = os.environ.get("HF_HOME", os.path.abspath("./.cache/hf"))
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Prefetching {repo_id} to {cache_dir} …")
    snapshot_download(
        repo_id=repo_id,
        local_dir=None,
        local_dir_use_symlinks=True,
        cache_dir=cache_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
