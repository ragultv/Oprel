# Binary Provenance & Checksum Verification Guide

How Oprel obtains the runtime binaries it uses, where they come from, where they are stored, and how supply-chain trust can be strengthened in the future. This guide is for operators, security reviewers, and anyone deploying Oprel in controlled environments who wants to understand the binary download pipeline.

---

## 1. Why Oprel Downloads Runtime Binaries

Oprel is a Python wrapper and orchestration layer around two native inference engines:

- **llama.cpp** for text generation, embeddings, and vision models.
- **stable-diffusion.cpp** for image generation.

These engines are written in C/C++ and distributed as pre-compiled binaries. Instead of requiring every user to compile them from source, Oprel downloads the correct binary for the current platform and GPU type on first use (or during installation, depending on the install path). This keeps the PyPI package small and avoids build-toolchain dependencies.

In short: Oprel downloads binaries because it relies on upstream native inference engines to do the actual model execution.

---

## 2. Which Upstream Projects Provide the Binaries

Oprel pulls binaries directly from the official GitHub releases of the upstream projects. No binaries are built or hosted by Oprel itself.

| Backend | Upstream project | Release source |
|---|---|---|
| **llama.cpp** | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `https://github.com/ggml-org/llama.cpp/releases/download/<build>/...` |
| **stable-diffusion.cpp** | [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | `https://github.com/leejet/stable-diffusion.cpp/releases/download/<tag>/...` |

The exact URLs are maintained in `oprel/runtime/binaries/registry.py`. Each entry records:

- `url` — the archive to download.
- `archive_type` — `zip` or `tar.gz`.
- `binary_name` — the upstream executable name inside the archive (for example, `llama-server` or `sd-cli`).
- `gpu_type` — `cpu`, `cuda`, `vulkan`, `metal`, or `rocm`.
- `dll_url` — optional separate archive for Windows CUDA runtime libraries.

Supported platform variants today include Linux x86_64 (CPU / Vulkan / ROCm for image generation), Windows x86_64 and arm64 (CPU / CUDA / Vulkan), and macOS x86_64 and arm64 (CPU / Metal).

---

## 3. Where Downloaded Binaries Are Stored

Downloaded binaries are placed under the user's cache directory by default:

```text
~/.cache/oprel/bin/
```

The path is controlled by `Config.binary_dir` in `oprel/core/config.py`. Binaries are isolated by backend, version, and GPU type to avoid collisions. A typical layout looks like:

```text
~/.cache/oprel/bin/
├── llama_cpp/
│   └── b9616/
│       ├── cpu/
│       │   ├── llama-server
│       │   └── oprel-backend
│       └── cuda/
│           ├── llama-server.exe
│           └── oprel-backend.exe
└── stable_diffusion_cpp/
    └── master_647_72e512a/
        └── cpu/
            ├── sd-cli
            └── oprel-backend
```

After extraction, Oprel also creates an `oprel-backend` copy of the executable so running processes are easier to identify.

---

## 4. Current Trust Model

Understanding the current model helps you decide whether additional controls are needed for your environment.

### What the current implementation does

- **HTTPS only.** All registry URLs use `https://github.com/...`. Downloads go through `urllib.request` with an SSL context (`oprel/runtime/binaries/installer.py`).
- **SSL verification is on by default.** You can disable it with `OPREL_SSL_NO_VERIFY=1` or `Config(ssl_verify=False)`, but this is intended only for corporate proxies with custom certificates.
- **Custom CA support.** You can point `Config.ssl_cert_file` at a corporate CA bundle.
- **Official upstream releases.** Binaries come from the public release pages of the two upstream projects, not from a third-party mirror.
- **Version pinning.** Default versions are set in `oprel/core/config.py` (`binary_version` and `image_binary_version`). You can override them through the `Config` object.

### What the current implementation does **not** do

As of the current source inspection:

- There is **no SHA256 checksum verification** of downloaded archives or extracted binaries.
- There is **no cryptographic signature verification** of the binaries.
- There is **no per-release manifest** published alongside the registry.
- Download failures are reported, but integrity mismatches cannot be detected because no expected digest is recorded.

This is not a vulnerability report; it is simply the current state. The sections below describe a practical way to improve supply-chain transparency if the project chooses to implement it.

---

## 5. Recommended Future Verification Model

A lightweight, fail-closed verification layer would give operators confidence that the binary they received is exactly the one the registry intended. The following is a proposed design, not a current feature.

### Core principles

1. **Publish a SHA256 manifest alongside the registry.** Each entry in the registry should include the expected SHA256 digest of the downloaded archive (and, optionally, the extracted binary).
2. **Per-platform, per-version, per-artifact checksums.** Because each platform downloads a different archive, the manifest must be keyed by `(backend, version, platform, accelerator)`.  When a platform also downloads a separate artifact — for example, the Windows CUDA runtime library archive referenced by `dll_url` — an optional `artifact` field (e.g. `"dll"`) distinguishes that archive from the main binary archive.
3. **Fail closed on mismatch.** If the computed digest does not match the manifest, delete the partial download and raise a clear error. Do not fall back to using the file.
4. **Optional signature verification later.** Once checksums are in place, the manifest itself can be signed (for example, with Sigstore or a project signing key) to protect against manifest tampering.
5. **Clear error messages.** Users should see exactly which file failed, what digest was expected, and what was received, plus instructions on how to skip downloads if they are managing binaries manually.

### Suggested manifest shape

The registry could be extended so that each platform entry includes an integrity block. A minimal addition would look like:

```json
{
  "backend": "llama.cpp",
  "version": "b9616",
  "platform": "Linux-x86_64",
  "accelerator": "cpu",
  "artifact": null,
  "url": "https://github.com/ggml-org/llama.cpp/releases/download/b9616/llama-b9616-bin-ubuntu-x64.tar.gz",
  "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
  "size": 12345678
}
```

Fields:

- `backend` — `llama.cpp` or `stable-diffusion.cpp`.
- `version` — the upstream build or tag.
- `platform` — the platform key used by the registry, for example `Linux-x86_64`.
- `accelerator` — `cpu`, `cuda`, `vulkan`, `metal`, or `rocm`.
- `artifact` — optional artifact qualifier.  `null` (or omitted) represents the main binary archive; `"dll"` represents the separate Windows CUDA runtime library archive.
- `url` — the HTTPS download URL.
- `sha256` — the expected SHA256 digest of the archive.
- `size` — optional byte size for an early sanity check.

### Where verification should run

The ideal place to add verification is in `oprel/runtime/binaries/installer.py`, inside `_safe_download` or immediately after it:

1. Download the archive to a temporary file.
2. Compute `hashlib.sha256()` of the temporary file.
3. Compare it to the manifest entry.
4. Only proceed with extraction if the digest matches.
5. On mismatch, raise `BinaryNotFoundError` with a message showing expected vs. actual digest.

### Optional: extract-once binary checksums

If upstream projects publish per-binary checksums in addition to archive checksums, Oprel could also verify the extracted executable before launching it. This adds another layer but requires upstream projects to publish those digests. Until they do, archive-level SHA256 verification is the most practical first step.

### Foundation implemented

A lightweight manifest and helper module — `oprel/runtime/binaries/integrity.py` — has been added as a first code-side step toward the model above. It provides:

- A `BinaryIntegrityEntry` dataclass matching the manifest shape described here, including an optional `artifact` field for separate archives such as the Windows CUDA DLL archive.
- An empty `BINARY_INTEGRITY_MANIFEST` placeholder (no digests populated yet).
- `get_integrity_entry()` lookup that returns `None` when no entry exists, with backward-compatible 4-tuple keys for the main archive and 5-tuple keys for artifact-qualified archives.
- `validate_sha256_format()` (case-insensitive), `compute_sha256()`, `verify_sha256()`, and `verify_size()` helpers.
- An `IntegrityMismatchError` exception and a `SizeMismatchError` exception for clear failure reporting.

The installer optionally verifies the main binary archive and, on the Windows CUDA path, the separate DLL archive before extraction. When a manifest entry includes an optional `size`, the file size is checked before SHA256 verification. The manifest remains intentionally empty, so runtime behavior is unchanged until real digests are populated.

---

## 6. Safe Operational Guidance for Users

Until automated checksum verification is implemented, you can use the following practices to reduce supply-chain risk.

### Skip automatic downloads in CI and controlled environments

Set the environment variable before installing Oprel:

```bash
export OPREL_SKIP_RUNTIME_DOWNLOAD=1
pip install oprel==<tested-version>
```

This prevents `setup.py` from downloading runtime binaries during installation. You can still fetch binaries later with `oprel setup runtime`, or place pre-verified binaries into `~/.cache/oprel/bin/` manually.

### Verify binaries manually if your threat model requires it

If you need stronger assurance:

1. Look up the URL for your platform in `oprel/runtime/binaries/registry.py`.
2. Download the archive yourself (for example, with `curl` or a browser).
3. Compute the SHA256 digest locally: `sha256sum <archive>` on Linux, `shasum -a 256 <archive>` on macOS, or `Get-FileHash -Algorithm SHA256 <archive>` on Windows.
4. Compare the digest against the upstream project's release notes or published checksum file, if available.
5. Extract the archive into the correct `~/.cache/oprel/bin/<backend>/<version>/<accelerator>/` directory.
6. Keep `OPREL_SKIP_RUNTIME_DOWNLOAD=1` for install-time workflows, and verify from source before relying on it for any runtime behavior.

### Pin Oprel versions in production

Use an exact version in requirements files and deployment scripts:

```bash
pip install oprel==<tested-version>
```

Pinning prevents unexpected registry changes when a new Oprel release updates default binary versions. It also makes audits easier because the registry contents are tied to a specific package version.

### Keep SSL verification enabled

Only disable SSL verification (`OPREL_SSL_NO_VERIFY=1` or `ssl_verify=False`) as a temporary workaround for corporate proxies. Prefer supplying a custom CA bundle via `Config.ssl_cert_file` so TLS certificate validation remains active.

### Do not copy binaries from untrusted sources

Avoid copying `~/.cache/oprel/bin/` between machines unless you trust the source and can verify the files. If you mirror binaries internally, maintain your own manifest of expected SHA256 digests and validate the files before Oprel uses them.

### Run backend processes with least privilege

The downloaded binaries are executed as separate processes. Run Oprel under a dedicated user account or container with limited permissions, and restrict access to the binary cache directory.

---

## 7. Relationship to Other Guides

- **[Safe Installation & Deployment Guide](install_hardening.md)** — covers `OPREL_SKIP_RUNTIME_DOWNLOAD`, SSL configuration, cache locations, server exposure, and the deployment checklist. Read it together with this guide when hardening an installation.
- **[Hardware & Deployment Guide](hardware_guide.md)** — explains how Oprel selects the CUDA, Vulkan, Metal, or CPU binary for your platform and how GPU detection influences the download.
- **[CLI Reference](cli_reference.md)** — documents `oprel setup runtime`, which downloads both backends on demand.

---

## 8. Summary

Oprel downloads official upstream release binaries for llama.cpp and stable-diffusion.cpp so users do not have to compile them. Today the trust model relies on HTTPS, SSL certificate verification, and the integrity of the upstream GitHub release pages. Automated checksum or signature verification of downloaded archives is not yet implemented. Adding a per-platform, per-version SHA256 manifest and failing closed on mismatch would be a practical next step for improving supply-chain transparency and operator confidence.
