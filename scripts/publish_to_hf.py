# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
import argparse
from pathlib import Path
from huggingface_hub import HfApi

def main() -> None:
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Target Hugging Face Repo ID (e.g. username/dataset_name)")
    parser.add_argument("--dataset_path", type=str, default="dataset-224", help="Local path to the dataset folder")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    parser.add_argument("--token", type=str, help="Hugging Face token (optional if logged in via CLI)")
    
    args = parser.parse_args()
    
    # helper to handle relative paths from the script execution location
    # verify dataset exists
    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset path '{args.dataset_path}' does not exist.")
        return

    api = HfApi(token=args.token)
    
    print(f"Creating repository {args.repo_id} (if not exists)...")
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    except Exception as e:
        print(f"Note on repo creation: {e}")
    
    print(f"Uploading files from {args.dataset_path} to {args.repo_id} using upload_large_folder...")
    try:
        api.upload_large_folder(
            folder_path=args.dataset_path,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print("Upload complete! 🚀")
        print(f"View your dataset at: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
