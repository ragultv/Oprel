# Downloader and Caching

The `oprel/downloader/` module ensures models are fetched securely, verified, and stored locally.

## Model Hub Resolution (`hub.py`)
Oprel primarily interfaces with the HuggingFace Hub.
1. **Alias Resolution**: Hardcoded dictionaries (`aliases.py`) map friendly names like `llama3-8b` to their exact HuggingFace repo counterparts (e.g., `QuantFactory/Meta-Llama-3-8B-Instruct-GGUF`).
2. **File Selection**: Because repositories contain dozens of GGUF files (representing different quantizations), `hub.py` filters the remote file tree to find the exact match requested by the `recommender.py`.
3. **Resumable Downloads**: Oprel implements HTTP `Range` headers. If a 10GB download is interrupted at 9GB, running `oprel pull` again will instantly resume from the 9GB mark.

## File Verification (`verification.py`)
GGUF files are massive binary blobs. Oprel calculates the `SHA-256` hash of the downloaded file and cross-references it with the metadata provided by the HuggingFace Hub API to guarantee against corruption during transit.

## Cache Indexing (`cache.py`)
All models are stored flat in `~/.cache/oprel/models`. Oprel maintains an ultra-fast indexing system (`metadata.py`) that stores JSON metadata alongside the binary files. This allows the CLI to run `oprel models` and list local files instantly without needing to boot up and parse gigabytes of data.
