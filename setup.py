"""
Setup script for Oprel SDK
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from pathlib import Path
import os

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # Run normal install
        install.run(self)

        # Allow CI/users to opt out of runtime downloads during install.
        if os.environ.get("OPREL_SKIP_RUNTIME_DOWNLOAD", "0").lower() in {"1", "true", "yes"}:
            print("Post-install: Skipping runtime backend download (OPREL_SKIP_RUNTIME_DOWNLOAD enabled).")
            return

        try:
            from oprel.core.config import Config
            from oprel.runtime.binaries.installer import ensure_binary

            config = Config()
            backends = [
                ("llama.cpp", config.binary_version),
                ("stable-diffusion.cpp", config.image_binary_version),
            ]

            print("Post-install: Downloading runtime backends (llama.cpp + stable-diffusion.cpp)...")
            for backend, version in backends:
                try:
                    binary_path = ensure_binary(
                        backend=backend,
                        version=version,
                        binary_dir=config.binary_dir,
                        config=config,
                    )
                    print(f"Post-install: {backend} is ready at {binary_path}")
                except Exception as backend_error:
                    print(f"Warning: Failed to download {backend} runtime: {backend_error}")
        except Exception as e:
            # Don't fail installation if the runtime setup path fails, just warn.
            print(f"Warning: Runtime backend post-install setup failed: {e}")
            print("You can still run: oprel setup runtime")


# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read version
version = {}
version_path = Path(__file__).parent / "oprel" / "version.py"
exec(version_path.read_text(), version)

setup(
    name="oprel",
    version=version["__version__"],
    author=version["__author__"],
    author_email=version["__email__"],
    
    description="Oprel is a high-performance Python library for running large language models locally. It provides a production-ready runtime with advanced memory management, hybrid offloading, and full multimodal support.Run LLMs locally with one line of Python. Ollama alternative with server mode, conversation memory, and 50+ model aliases. The SQLite of AI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Skyroot-Solutions/Oprel",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "huggingface-hub>=0.20.0",  
        "psutil>=5.9.0",
        "requests>=2.31.0",
        "pydantic>=2.10.0",
        "rich>=13.0.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "aiofiles>=24.1.0",
        "python-multipart>=0.0.20",
        "starlette>=0.41.3",
        "chromadb>=0.5.0",
        "rank_bm25>=0.2.2",
        "google-genai>=2.0.0",
        "groq>=0.9.0",
    ],
    extras_require={
        "local": ["torch>=2.1.0"],
        "cuda": ["torch>=2.1.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "oprel=oprel.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Typing :: Typed",
        "Environment :: Console",
        "Environment :: GPU",
        "Natural Language :: English",
    ],
    keywords="llm local-llm ollama ollama-alternative llama3 qwencoder gemma mistral gguf llama.cpp python-llm local-ai offline-ai conversational-ai text-generation model-server ai-runtime machine-learning privacy edge-ai",
    project_urls={
        "Documentation": "https://docs.oprel.Skyroot-Solutions.com",
        "Source": "https://github.com/Skyroot-Solutions/Oprel",
        "Bug Reports": "https://github.com/Skyroot-Solutions/Oprel/issues",
        "Changelog": "https://github.com/Skyroot-Solutions/Oprel/releases",
    },
    cmdclass={
        'install': PostInstallCommand,
    },
)
