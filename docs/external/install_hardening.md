# Safe Installation & Deployment Guide

Practical guidance for installing, configuring, and deploying Oprel in local, CI, and controlled environments. This guide covers runtime downloads, cache locations, server exposure, API key handling, and optional features — helping you understand what Oprel does on your machine so you can deploy it confidently.

---

## 1. Overview

Oprel can be used in several ways:

- **Local development** — install via `pip`, run models on your own machine.
- **CI/CD pipelines** — install in automated environments with runtime downloads skipped.
- **Controlled deployments** — run the server behind a reverse proxy with TLS and authentication.

This guide explains what happens during installation and runtime, where files are stored, how network exposure works, and what to consider for production use. It is not a security audit — it is practical deployment guidance based on source inspection.

---

## 2. Installation Modes

### Standard pip install

```bash
pip install oprel==0.7.1
```

Depending on the installation path, Oprel may invoke setup.py post-install behavior that downloads runtime binaries (llama.cpp and stable-diffusion.cpp) for your platform. These are placed in `~/.cache/oprel/bin/`.

### Controlled / CI install (skip runtime downloads)

Set the `OPREL_SKIP_RUNTIME_DOWNLOAD` environment variable before installing:

```bash
OPREL_SKIP_RUNTIME_DOWNLOAD=1 pip install oprel==0.7.1
```

This skips the automatic binary download during installation. You can download runtimes later with `oprel setup runtime` or let them download on first use.

### Manual runtime setup

After installing, you can explicitly download runtimes:

```bash
oprel setup runtime
```

This downloads both llama.cpp and stable-diffusion.cpp binaries for your platform. The command is confirmed in source (`oprel/cli/main.py:1055-1056`).

### Offline / air-gapped preparation

For environments without internet access:

1. Install Oprel with `OPREL_SKIP_RUNTIME_DOWNLOAD=1`.
2. On a connected machine, download the binaries and model files to the appropriate cache directories.
3. Copy `~/.cache/oprel/` to the target machine.
4. For OCR, pre-install `paddlepaddle` and `paddleocr` packages and copy `~/.cache/oprel/ocr/`.

Note: Oprel does not provide a built-in offline bundle. You will need to manually transfer cache directories.

---

## 3. Runtime Downloads and Local Storage

Oprel downloads binaries and models on first use. All downloaded content is stored under `~/.cache/oprel/` by default.

### Confirmed cache paths

| Directory | Contents | Verified in source |
|---|---|---|
| `~/.cache/oprel/models/` | Downloaded GGUF model files | `oprel/core/config.py:42` |
| `~/.cache/oprel/bin/` | llama-server and sd-cli binaries | `oprel/core/config.py:43` |
| `~/.cache/oprel/ocr/` | PaddleOCR models (detection, recognition, classification) | `oprel/server/services/ocr_service.py:32` |
| `~/.cache/oprel/models/chat_history.db` | SQLite database for conversations, provider configs, settings, and inference logs | `oprel/server/db.py:10` |

An additional file is stored at `~/.oprel/hardware_profile.json` for hardware profiling (`oprel/telemetry/profiler.py:5`).

### OPREL_HOME scope

The `OPREL_HOME` environment variable controls the **knowledge directory only**:

```python
# oprel/knowledge/config.py
OPREL_HOME = Path(os.environ.get("OPREL_HOME", Path.home() / ".cache" / "oprel"))
KNOWLEDGE_DIR = OPREL_HOME / "knowledge"
```

`OPREL_HOME` does **not** control the model cache, binary cache, OCR cache, or database paths. Those are hardcoded to `~/.cache/oprel/` via `Config.cache_dir` and `Config.binary_dir` in `oprel/core/config.py:42-43`, which default to `~/.cache/oprel/models` and `~/.cache/oprel/bin` respectively.

---

## 4. API Key Handling

Oprel supports cloud provider API keys (OpenAI, Groq, Google Gemini, NVIDIA NIM, OpenRouter, and custom OpenAI-compatible endpoints).

### How keys are supplied

1. **Environment variables** — set keys like `OPENAI_API_KEY` in your shell or `.env` file.
2. **Oprel Studio WebUI** — paste keys into the Settings > AI Providers modal.
3. **`.env` file** — place in your project root or `~/.oprel/` directory.

### Storage

When keys are saved via the WebUI, they are stored in the local SQLite database at `~/.cache/oprel/models/chat_history.db` in the `provider_configs` table. The source code comments explicitly note: "api_key is stored as-is; for production deployments consider encrypting at rest" (`oprel/server/db.py:108`).

API keys passed via environment variables are not persisted to disk by Oprel itself.

### Recommendations

- Keep `.env` files out of version control.
- Restrict file permissions on the SQLite database: `chmod 600 ~/.cache/oprel/models/chat_history.db`.
- On shared machines, be aware that any user with read access to the database can read stored API keys.
- For production, consider encrypting the database at rest or using environment variables instead of WebUI-stored keys.

---

## 5. Server Exposure

### Default binding

The Oprel server (daemon) binds to `127.0.0.1` (localhost) on port `11435` by default:

```python
# oprel/server/app.py:196
def run_server(host: str = "127.0.0.1", port: int = 11435):
```

This means only local applications can reach the server. Remote access is blocked by default.

The inference backends (llama-server) use a separate port range (`54321–54420`) configured in `oprel/core/config.py:75-76`.

### Network binding

When you pass `--host 0.0.0.0` to `oprel serve`, the server becomes accessible to any device on the network. The CLI defaults to `127.0.0.1` (`oprel/cli/main.py:775,792`).

### CORS configuration

The server uses wildcard CORS (`allow_origins=["*"]`) with credentials enabled (`oprel/server/app.py:156-163`). This is standard for local development but should be restricted for production deployments.

### Authentication

For remote deployments, do not rely on the default local API exposure alone. Use an external authentication layer such as a reverse proxy with auth. The OpenAI-compatible example uses `api_key="not-needed"`, so verify the authentication behavior you need before exposing the service beyond localhost.

### Recommendations for remote access

- Put Oprel behind a reverse proxy (nginx, Caddy) with TLS.
- Add authentication at the proxy layer (basic auth, client certificates, etc.).
- Restrict CORS origins to your application's domain.
- Use a firewall to limit access to trusted IPs.

---

## 6. SSL/TLS and Corporate Proxies

Oprel supports SSL configuration for binary downloads via the `Config` class (`oprel/core/config.py:138-144`):

| Setting | Type | Default | Description |
|---|---|---|---|
| `ssl_verify` | `bool` | `True` | Verify SSL certificates when downloading binaries |
| `ssl_cert_file` | `Path \| None` | `None` | Custom CA certificate bundle for corporate proxies |

### Environment variable

Set `OPREL_SSL_NO_VERIFY=1` to disable SSL verification for binary downloads. This is useful behind corporate proxies with self-signed certificates.

### Config object

```python
from oprel import Model
from oprel.core.config import Config

config = Config(ssl_verify=False)
model = Model("gemma3-1b", config=config)
```

### Recommendations

- Disabling SSL verification should be temporary and limited to controlled environments.
- Prefer providing a custom CA certificate via `ssl_cert_file` over disabling verification entirely.
- See `examples/ssl_configuration.py` for complete examples including corporate proxy setup.

---

## 7. Optional OCR Behavior

Oprel includes a built-in OCR pipeline powered by PaddleOCR. On first use, the OCR setup process:

1. Installs `paddlepaddle` via pip (`oprel/server/services/ocr_service.py:136-137`).
2. Installs `paddleocr` via pip (`oprel/server/services/ocr_service.py:145-146`).
3. Downloads OCR models to `~/.cache/oprel/ocr/` (`oprel/server/services/ocr_service.py:152-154`).

This happens automatically when you click "Download OCR Models" in Oprel Studio or trigger OCR via the API.

### Recommendations for controlled environments

- Pre-install `paddlepaddle` and `paddleocr` in your Python environment before deploying.
- Pre-download OCR models to `~/.cache/oprel/ocr/` and copy to target machines.
- If you do not need OCR, no action is required — the packages are only installed on first use.

---

## 8. Environment Variables

| Variable | Effect | Verified in source |
|---|---|---|
| `OPREL_SKIP_RUNTIME_DOWNLOAD` | Set to `1`, `true`, or `yes` to skip runtime binary downloads during `pip install` | `setup.py:17` |
| `OPREL_SSL_NO_VERIFY` | Set to `1` to disable SSL verification for binary downloads | `examples/ssl_configuration.py:21`, `hardware_guide.md` |
| `OPREL_HOME` | Overrides the knowledge directory base path (default: `~/.cache/oprel`). Does **not** control model, binary, OCR, or database paths | `oprel/knowledge/config.py:8` |

---

## 9. Deployment Checklist

- [ ] **Use a virtual environment** — avoid global pip installs; use `python -m venv` or `conda`.
- [ ] **Understand what gets downloaded** — runtime binaries and model files can be large depending on the selected backend and model, and are cached under `~/.cache/oprel/`.
- [ ] **Keep API keys out of git** — use environment variables or `.env` files excluded from version control.
- [ ] **Keep the server on localhost** — only bind to `0.0.0.0` if protected by a reverse proxy and firewall.
- [ ] **Use reverse proxy / auth / firewall** for remote access — Oprel has no built-in authentication.
- [ ] **Review cache and database file permissions** — especially on shared machines where API keys are stored in the SQLite database.
- [ ] **Pin versions in production / CI** — use the exact Oprel version you tested instead of an unpinned install to avoid unexpected upgrades.
- [ ] **Pre-install OCR dependencies** in offline/air-gapped environments — `paddlepaddle` and `paddleocr` are installed on first OCR use.

---

## 10. Related Docs

- [Hardware & Deployment Guide](hardware_guide.md) — GPU acceleration, RAM/VRAM requirements, hardware tiers.
- [Binary Provenance & Checksum Verification Guide](binary_provenance.md) — where runtime binaries come from, current trust model, and a proposed verification manifest.
- [Cloud Providers](cloud_providers.md) — configuring external AI providers and API keys.
- [CLI Reference](cli_reference.md) — all Oprel CLI commands and options.
