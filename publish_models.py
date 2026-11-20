"""Publish OpenVINO IR models from models/ to the m4gshm HuggingFace account.

Usage:
    python publish_models.py [--models NAME ...] [--dry-run]

Idempotent: re-running uploads only changed files; nothing is deleted.
Extend MODELS below to publish more models.
"""
import argparse
import os
import sys

from huggingface_hub import HfApi, whoami

sys.path.insert(0, "models")
from gen_readmes import ensure_readme  # pylint: disable=wrong-import-position,import-error

ORG = "m4gshm"
ROOT = "models"
# Models to publish: folder names under models/, uploaded as m4gshm/<folder>.
MODELS = [
    "Qwen2.5-Coder-0.5B-int4-sym-g128-se-awq",
    "Qwen2.5-Coder-0.5B-int8-sym",
    "Qwen2.5-Coder-0.5B-nf4-sym-g-1-se-awq",
    "Qwen2.5-Coder-1.5B-int4-sym-g128-se-awq",
    "Qwen2.5-Coder-1.5B-int8-sym",
    "Qwen2.5-Coder-1.5B-nf4-sym-g-1-se-awq",
    "Qwen2.5-Coder-3B-int4-sym-g128-se-awq",
    "Qwen2.5-Coder-3B-int8-sym",
    "Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq",
]


def check_auth() -> bool:
    try:
        info = whoami()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"ERROR: no HuggingFace credentials ({exc}).\n"
              "Set HF_TOKEN or run `huggingface-cli login` with write access to "
              f"https://huggingface.co/{ORG}.")
        return False
    print(f"Authenticated as {info.get('name')}")
    return True


def publish(api: HfApi, folder: str, dry_run: bool) -> bool:
    """Generate the model card if needed and upload the folder to HF."""
    d = os.path.join(ROOT, folder)
    if not os.path.isdir(d):
        print(f"SKIP missing folder: {d}")
        return False
    readme = ensure_readme(d)
    print(f"{'wrote' if readme else 'kept existing'} README for {folder}")
    if dry_run:
        print(f"DRY-RUN would upload {d} -> {ORG}/{folder}")
        return True
    api.create_repo(f"{ORG}/{folder}", repo_type="model", private=False, exist_ok=True)
    # A card already on HF wins: avoid weak cosmetic diffs on re-runs.
    remote = api.list_repo_files(f"{ORG}/{folder}", repo_type="model")
    ignore = ["README.md"] if "README.md" in remote else None
    api.upload_folder(folder_path=d, repo_id=f"{ORG}/{folder}", repo_type="model",
                      commit_message=f"Update {folder}", ignore_patterns=ignore)
    print(f"OK {ORG}/{folder}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=MODELS,
                        help="subset of MODELS to publish (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="generate READMEs but do not upload")
    args = parser.parse_args()

    if not check_auth():
        return 2
    api = HfApi()
    failed = [m for m in args.models if not publish(api, m, args.dry_run)]
    if failed:
        print("FAILED models:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
