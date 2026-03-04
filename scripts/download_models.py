import os
import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="BAAI/bge-small-en-v1.5")
snapshot_download(repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2")
print(" Done!")

print("\nAll models downloaded successfully! You can now run `streamlit run app.py`.")
