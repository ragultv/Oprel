"""
Main CLI entry point for Oprel SDK.
"""
import argparse
import sys
import os
import signal
from pathlib import Path

# Important: Imports are deferred inside commands where possible to speed up startup
from oprel import __version__
from oprel.utils.logging import set_log_level, get_logger

# Import decoupled command modules
from .text import cmd_chat, cmd_generate, cmd_run
from .image import cmd_gen_image, cmd_setup_image, cmd_setup_runtimes
from .vision import cmd_vision
from .embed import cmd_embed
from .knowledge import cmd_index, cmd_knowledge_search, cmd_knowledge_sync

logger = get_logger(__name__)


def _unsupported_feature(feature: str) -> int:
    print(
        f"{feature} is not available in this build.\n"
        "This Oprel build currently exposes text, vision, embeddings, and image generation only."
    )
    return 1


def handle_sigint(signum, frame):
    """Handle Ctrl+C globally"""
    # Just exit, allowing cleanup in finally blocks if any
    sys.exit(130)

signal.signal(signal.SIGINT, handle_sigint)


def cmd_info(args: argparse.Namespace) -> int:
    """Show system information"""
    from oprel.telemetry.hardware import get_system_info
    import json
    
    info = get_system_info()
    print(f"Oprel SDK v{__version__}")
    print("-" * 30)
    print(f"OS: {info['os']} {info['os_release']}")
    print(f"Python: {info['python_version']}")
    print(f"CPU: {info['cpu_brand']} ({info['cpu_cores']} cores)")
    print(f"RAM: {info['ram_total_gb']:.1f} GB")
    
    if info['gpu_count'] > 0:
        print(f"GPU: {info['gpu_name']} ({info['vram_total_gb']:.1f} GB VRAM)")
    else:
        print("GPU: None detected")
        
    print("-" * 30)
    return 0



def cmd_cache_list(args: argparse.Namespace) -> int:
    """List cached models using HuggingFace Hub cache scanning"""
    from oprel.core.config import Config
    from oprel.downloader.aliases import MODEL_ALIASES
    from huggingface_hub import scan_cache_dir
    
    config = Config()
    cache_dir = config.cache_dir
    
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return 0
        
    try:
        scan = scan_cache_dir(cache_dir)
    except Exception as e:
        print(f"Error scanning cache: {e}")
        return 1
    
    # helper to find alias from repo_id
    repo_to_alias = {}
    for alias, r_id in MODEL_ALIASES.items():
        # Clean the repo ID from alias dict (sometime includes :filename)
        clean_repo_id = r_id.split(':')[0]
        if clean_repo_id not in repo_to_alias or len(alias) < len(repo_to_alias[clean_repo_id]):
            repo_to_alias[clean_repo_id] = alias

    print(f"{'Alias / Model ID':<40} {'Size':<10} {'Refs'}")
    print("-" * 75)
    
    total_size = 0
    repos = sorted(scan.repos, key=lambda r: r.size_on_disk, reverse=True)
    
    if not repos:
        print("No models found in cache.")
        return 0

    for repo in repos:
        repo_id = repo.repo_id
        size_gb = repo.size_on_disk / (1024**3)
        total_size += size_gb
        
        display_name = repo_to_alias.get(repo_id, repo_id)
        if len(display_name) > 38:
            display_name = display_name[:35] + "..."
            
        # Count variations/revisions
        refs = len(repo.refs)
        print(f"{display_name:<40} {size_gb:.2f} GB   {refs} refs")
        
    print("-" * 75)
    print(f"Total: {len(repos)} repositories, {total_size:.2f} GB")
    return 0


def cmd_cache_delete(args: argparse.Namespace) -> int:
    """Delete specific model from cache (supports Alias or Repo ID)"""
    from oprel.core.config import Config
    from oprel.downloader.aliases import MODEL_ALIASES
    from huggingface_hub import scan_cache_dir
    import shutil
    
    config = Config()
    model_name_input = args.model_name
    
    # Resolve alias to repo_id if possible
    target_repo_id = model_name_input
    if model_name_input in MODEL_ALIASES:
        target_repo_id = MODEL_ALIASES[model_name_input].split(':')[0]
        print(f"Resolved alias '{model_name_input}' -> '{target_repo_id}'")
    
    try:
        scan = scan_cache_dir(config.cache_dir)
    except Exception as e:
        print(f"Error scanning cache: {e}")
        return 1
    
    # Find matching repo
    found_repos = [r for r in scan.repos if r.repo_id == target_repo_id]
    
    if not found_repos:
        # Try fuzzy match if exact match failed
        found_repos = [r for r in scan.repos if target_repo_id.lower() in r.repo_id.lower()]
    
    if not found_repos:
        print(f"❌ Model '{model_name_input}' not found in cache.")
        return 1
        
    if len(found_repos) > 1:
        print(f"Found multiple matching repositories:")
        for i, repo in enumerate(found_repos):
            print(f"  {i+1}. {repo.repo_id} ({repo.size_on_disk / (1024**3):.2f} GB)")
            
        selection = input("\nEnter number to delete (or 'q' to cancel): ")
        if selection.lower() == 'q':
            return 0
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(found_repos):
                found_repos = [found_repos[idx]]
            else:
                print("Invalid selection")
                return 1
        except ValueError:
            print("Invalid input")
            return 1
            
    # Delete the repo
    repo_to_delete = found_repos[0]
    print(f"Deleting {repo_to_delete.repo_id} ({repo_to_delete.size_on_disk / (1024**3):.2f} GB)...")
    
    try:
        # scan_cache_dir returns repo information including path
        # repo_to_delete.repo_path points to the models--... directory
        path_to_delete = repo_to_delete.repo_path
        if path_to_delete.exists():
            shutil.rmtree(path_to_delete)
            print(f"✓ Successfully deleted {repo_to_delete.repo_id}")
            return 0
        else:
            print(f"Error: Path {path_to_delete} does not exist")
            return 1
    except Exception as e:
        print(f"❌ Error deleting cache: {e}")
        return 1


def cmd_cache_clear(args: argparse.Namespace) -> int:
    """Clear all cached models."""
    from oprel.core.config import Config
    import shutil

    config = Config()
    cache_dir = config.cache_dir

    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return 0

    if not getattr(args, "yes", False):
        confirm = input(f"Delete all cached models in '{cache_dir}'? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancelled.")
            return 0

    try:
        # Remove only model cache entries and keep cache root intact.
        removed = 0
        for entry in cache_dir.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed += 1
            except Exception as e:
                print(f"Warning: failed to remove {entry.name}: {e}")

        print(f"✓ Cache cleared ({removed} item(s) removed)")
        return 0
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return 1



def cmd_list_models(args: argparse.Namespace) -> int:
    """List all available model aliases"""
    from oprel.downloader.aliases import (
        list_models_by_category,
        get_categories,
        get_category_info,
        MODEL_ALIASES
    )
    
    # Get category filter if provided
    category_filter = getattr(args, 'category', None)
    
    try:
        # Get models (filtered or all)
        categorized_models = list_models_by_category(category_filter)
        
        # Count total models
        total_count = sum(len(models) for models in categorized_models.values())
        
        # Print header
        if category_filter:
            cat_info = get_category_info(category_filter)
            print(f"\n{cat_info['icon']} {cat_info['name']} Models ({total_count} total)")
            print(f"   {cat_info['description']}\n")
        else:
            print(f"\nAvailable Models ({total_count} total across {len(categorized_models)} categories)\n")
        
        # Display models by category
        for category, models in categorized_models.items():
            cat_info = get_category_info(category)
            
            if not category_filter:
                print(f"{cat_info['icon']} {cat_info['name']} ({len(models)} models)")
                print(f"   {cat_info['description']}")
            
            # Sort models by alias
            for alias in sorted(models.keys()):
                repo_id = models[alias]
                source = repo_id.split("/")[0]
                model_name = repo_id.split("/")[-1]
                print(f"   {alias:25} → {source}/{model_name[:40]}")
            print()
        
        # Show available categories if listing all
        if not category_filter:
            print("Filter by category:")
            print("   oprel list-models --category <category>")
            print(f"   Categories: {', '.join(get_categories())}")
        
        print("\nUsage: oprel run <alias> \"your prompt\"")
        return 0
        
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable categories: {', '.join(get_categories())}")
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search for model aliases"""
    from oprel.downloader.aliases import search_aliases, MODEL_ALIASES
    
    matches = search_aliases(args.query)
    
    if not matches:
        print(f"No models found matching '{args.query}'")
        return 1
    
    print(f"Models matching '{args.query}':\n")
    for alias in matches:
        gguf_id = MODEL_ALIASES.get(alias, "")
        print(f"  {alias:20} -> {gguf_id}")
    
    print(f"\nUsage: oprel run {matches[0]} \"your prompt\"")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    """Download a model without running it."""
    from oprel.downloader.hub import download_model as download_gguf
    from oprel.downloader.aliases import MODEL_ALIASES, get_model_category
    from oprel.downloader.image_hub import resolve_image_model_assets

    model_id = args.model
    logger.info(f"Pulling model: {model_id}")

    if get_model_category(model_id) == "text-to-image":
        try:
            target_model = MODEL_ALIASES.get(model_id, model_id)
            assets = resolve_image_model_assets(target_model)
            print(f"Downloaded image model assets for: {target_model}")
            print(f"Primary asset: {assets.primary_path}")
            return 0
        except Exception as exc:
            print(f"Error downloading model: {exc}")
            return 1

    target_model = MODEL_ALIASES.get(model_id, model_id)
    print(f"Downloading model: {target_model}...")
    try:
        download_gguf(target_model)
        if model_id in MODEL_ALIASES:
            print(f"Successfully pulled {model_id}")
        return 0
    except Exception as exc:
        print(f"Error downloading model: {exc}")
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the oprel daemon server"""
    try:
        from oprel.server.daemon import run_server
        
        port = args.port
        host = args.host
        
        print(f"Starting Oprel daemon server...")
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        print()
        
        run_server(host=host, port=port)
        return 0
        
    except ImportError as e:
        logger.error(
            "Server dependencies not installed. "
            "Install with: pip install oprel[server]"
        )
        logger.error(f"Details: {e}")
        return 1
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1


def cmd_start(args: argparse.Namespace) -> int:
    """Start the server and open the Web UI"""
    import webbrowser
    import threading
    import time
    
    port = getattr(args, 'port', 11435)
    host = getattr(args, 'host', '127.0.0.1')
    
    # Function to open browser after a delay
    def open_browser():
        time.sleep(2)  # Wait for server to start
        url = f"http://{host}:{port}/gui/"
        print(f"Opening Oprel Studio at {url}...")
        webbrowser.open(url)
        
    # Start browser thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start server (this blocks)
    return cmd_serve(args)



def cmd_models(args: argparse.Namespace) -> int:
    """List available models (downloaded and loaded)"""
    import requests
    from oprel.core.config import Config
    from oprel.downloader.aliases import MODEL_ALIASES
    from huggingface_hub import scan_cache_dir
    
    config = Config()
    server_url = f"http://{args.host}:{args.port}"
    
    # Get loaded models from server
    loaded_models = {}
    server_online = False
    
    try:
        response = requests.get(f"{server_url}/models", timeout=2)
        if response.status_code == 200:
            server_online = True
            for m in response.json():
                loaded_models[m['model_id']] = m
    except requests.RequestException:
        pass
        
    # Scan local cache
    cached_models = []
    
    try:
        if config.cache_dir.exists():
            scan = scan_cache_dir(config.cache_dir)
            
            # Helper to map repo_id -> alias
            repo_to_alias = {}
            for alias, r_id in MODEL_ALIASES.items():
                clean_repo_id = r_id.split(':')[0]
                if clean_repo_id not in repo_to_alias or len(alias) < len(repo_to_alias[clean_repo_id]):
                    repo_to_alias[clean_repo_id] = alias

            for repo in sorted(scan.repos, key=lambda r: r.size_on_disk, reverse=True):
                repo_id = repo.repo_id
                size_gb = repo.size_on_disk / (1024**3)
                
                # Try to map to loaded model if possible
                is_loaded = repo_id in loaded_models
                
                # If using alias in loaded_models, check that too
                alias = repo_to_alias.get(repo_id, repo_id)
                if alias in loaded_models:
                    is_loaded = True
                    
                cached_models.append({
                    "name": alias,
                    "id": repo_id,
                    "size": f"{size_gb:.1f} GB",
                    "loaded": is_loaded
                })
    except Exception as e:
        logger.warning(f"Failed to scan cache: {e}")

    # Display results
    print(f"{'Model':<40} {'Size':<10} {'Status'}")
    print("-" * 70)
    
    if not cached_models:
        print("No models found locally.")
        if not server_online:
            print("Server is offline.")
        return 0
        
    for m in cached_models:
        status = "Loaded 🟢" if m['loaded'] else ""
        print(f"{m['name']:<40} {m['size']:<10} {status}")
        
    print("-" * 70)
    
    if not server_online:
        print(f"Note: Server at {server_url} is offline (cannot show loaded status)")
        
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop the oprel daemon server and all backend processes"""
    import requests
    import psutil
    import time
    
    server_url = f"http://{args.host}:{args.port}"
    port = args.port
    stopped_server = False
    stopped_backends = False
    
    # Step 1: Try graceful shutdown via API (this will unload all models and exit)
    try:
        print("Requesting graceful shutdown...")
        response = requests.post(f"{server_url}/shutdown", timeout=5)
        if response.status_code == 200:
            print("  ✓ Shutdown signal sent")
            # Wait a moment for the server to clean up
            time.sleep(2)
            
            # Check if it actually stopped
            if not _is_port_in_use(port):
                print("  ✓ Daemon stopped gracefully")
                stopped_server = True
                # If graceful shutdown worked, backend processes should be cleaned up too
                stopped_backends = True
                print("\n✓ All Oprel processes stopped successfully")
                return 0
    except Exception as e:
        logger.debug(f"Graceful shutdown failed: {e}")
    
    # Step 2: If graceful shutdown failed, unload models manually
    try:
        response = requests.get(f"{server_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            loaded_models = [m for m in models if m.get("loaded", False)]
            if loaded_models:
                print(f"Unloading {len(loaded_models)} model(s)...")
                for model in loaded_models:
                    model_id = model["model_id"]
                    import urllib.parse
                    encoded_id = urllib.parse.quote(model_id, safe="")
                    try:
                        unload_response = requests.delete(f"{server_url}/unload/{encoded_id}", timeout=30)
                        if unload_response.status_code == 200:
                            print(f"  ✓ Unloaded: {model_id}")
                        else:
                            print(f"  ✗ Failed to unload: {model_id}")
                    except Exception as e:
                        logger.debug(f"Failed to unload {model_id}: {e}")
    except Exception as e:
        logger.debug(f"Could not communicate with server to unload models: {e}")
    
    # Step 3: Find and kill the daemon server process
    try:
        server_pid = None
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                server_pid = conn.pid
                break
        
        if server_pid:
            try:
                process = psutil.Process(server_pid)
                process_name = process.name()
                print(f"Stopping Oprel daemon (PID: {server_pid})...")
                
                # Terminate gracefully
                process.terminate()
                try:
                    process.wait(timeout=5)
                    print(f"  ✓ Daemon stopped")
                    stopped_server = True
                except psutil.TimeoutExpired:
                    print(f"  Process didn't stop gracefully, forcing...")
                    process.kill()
                    process.wait()
                    print(f"  ✓ Daemon killed")
                    stopped_server = True
                    
                # Give the port time to be released
                time.sleep(0.5)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Could not stop daemon process {server_pid}: {e}")
        else:
            if not stopped_server:
                print("Oprel daemon is not running")
            
    except Exception as e:
        logger.debug(f"Error finding/stopping daemon: {e}")
    
    # Step 4: Kill any orphaned backend processes
    # These might be left over if the daemon crashed or was forcefully killed
    try:
        killed_backends = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_info = proc.info
                proc_name = proc_info.get('name', '').lower()
                cmdline = proc_info.get('cmdline', [])
                
                # Look for oprel-backend processes (our renamed binaries)
                # or fallback to llama-server if running older version
                is_backend = False
                
                # Primary check: look for our branded "oprel-backend" process
                if proc_name and 'oprel-backend' in proc_name:
                    is_backend = True
                # Fallback: look for llama-server from our binary directory
                elif proc_name and any(x in proc_name for x in ['llama-server', 'llama_server', 'llama-cpp']):
                    # Additional check: only kill if it's from oprel's binary directory
                    # This avoids killing user's own llama-server instances
                    if cmdline:
                        cmdline_str = ' '.join(cmdline).lower()
                        from oprel.core.config import Config
                        config = Config()
                        binary_dir_str = str(config.binary_dir).lower()
                        if binary_dir_str in cmdline_str:
                            is_backend = True
                
                if is_backend:
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)
                        killed_backends.append((proc_info['pid'], proc_info['name']))
                    except psutil.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                        killed_backends.append((proc_info['pid'], proc_info['name']))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if killed_backends:
            print(f"Stopped {len(killed_backends)} backend processes:")
            for pid, name in killed_backends:
                print(f"  ✓ {name} (PID: {pid})")
            stopped_backends = True
            
    except Exception as e:
        logger.debug(f"Error cleaning up backend processes: {e}")
    
    # Summary
    if stopped_server or stopped_backends:
        print("\n✓ All Oprel processes stopped successfully")
        return 0
    else:
        print("\nNo Oprel processes were running")
        return 0




def _is_port_in_use(port: int) -> bool:
    """Helper to check if a port is in use"""
    try:
        import psutil
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                return True
        return False
    except:
        return False


def cmd_recommend(args: argparse.Namespace) -> int:
    """Show model recommendations based on system hardware"""
    from oprel.recommendations import show_recommendations
    show_recommendations()
    return 0


def main() -> int:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="oprel",
        description="Oprel SDK - Local-first AI runtime",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"oprel {__version__}",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Recommend command
    subparsers.add_parser(
        "recommend",
        help="Analyze hardware and suggest best models"
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("model", help="Model ID (e.g., TheBloke/Llama-2-7B-GGUF)")
    chat_parser.add_argument("--quantization", help="Quantization level (Q4_K_M, Q8_0, etc.)")
    chat_parser.add_argument("--max-memory", type=int, help="Max memory in MB")
    chat_parser.add_argument("--stream", action="store_true", default=True, help="Stream responses")
    chat_parser.add_argument("--system", help="System prompt")
    chat_parser.add_argument("--thinking", action="store_true", help="Enable deep thinking (reasoning) mode")
    chat_parser.add_argument(
        "--no-server",
        action="store_true",
        help="Force direct mode (don't use persistent server)"
    )
    chat_parser.add_argument("--allow-low-quality", action="store_true", help="Allow low-quality quantizations like Q2_K")
    chat_parser.add_argument("--rag", action="store_true", help="Enable Retrieval-Augmented Generation")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from prompt")
    gen_parser.add_argument("model", help="Model ID")
    gen_parser.add_argument("prompt", help="Input prompt")
    gen_parser.add_argument("--quantization", help="Quantization level")
    gen_parser.add_argument("--max-memory", type=int, help="Max memory in MB")
    gen_parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--stream", action="store_true", help="Stream response")
    gen_parser.add_argument("--thinking", action="store_true", help="Enable deep thinking (reasoning) mode")
    gen_parser.add_argument(
        "--no-server",
        action="store_true",
        help="Force direct mode (don't use persistent server)"
    )
    gen_parser.add_argument("--rag", action="store_true", help="Enable Retrieval-Augmented Generation")

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the oprel daemon server for persistent model caching"
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=11435,
        help="Port to listen on (default: 11435)"
    )

    # Start command (Serve + Browser)
    start_parser = subparsers.add_parser(
        "start",
        help="Start Oprel Studio (Web UI)"
    )
    start_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=11435,
        help="Port to listen on (default: 11435)"
    )

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a model (single-shot or interactive mode)"
    )
    run_parser.add_argument("model", help="Model ID")
    run_parser.add_argument("prompt", nargs="?", default=None, help="Input prompt (omit for interactive mode)")
    run_parser.add_argument("--quantization", help="Quantization level")
    run_parser.add_argument("--max-memory", type=int, help="Max memory in MB")
    run_parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens to generate")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    run_parser.add_argument("--stream", action="store_true", default=True, help="Stream response")
    run_parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    run_parser.add_argument("--thinking", action="store_true", help="Enable deep thinking (reasoning) mode")
    run_parser.add_argument(
        "--no-server",
        action="store_true",
        help="Force direct mode (don't use persistent server)"
    )
    run_parser.add_argument("--allow-low-quality", action="store_true", help="Allow low-quality quantizations like Q2_K")
    run_parser.add_argument("--rag", action="store_true", help="Enable Retrieval-Augmented Generation")
    
    # Models command
    models_parser = subparsers.add_parser(
        "models",
        help="List models loaded in the server"
    )
    models_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    models_parser.add_argument(
        "--port",
        type=int,
        default=11435,
        help="Server port (default: 11435)"
    )

    # Stop command
    stop_parser = subparsers.add_parser(
        "stop",
        help="Request server to unload all models"
    )
    stop_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    stop_parser.add_argument(
        "--port",
        type=int,
        default=11435,
        help="Server port (default: 11435)"
    )

    # Info command
    subparsers.add_parser("info", help="Show system information")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup additional features")
    setup_subparsers = setup_parser.add_subparsers(dest="setup_command")
    setup_subparsers.add_parser("runtime", help="Download and prepare llama.cpp + stable-diffusion.cpp backends")
    setup_subparsers.add_parser("image", help="Download and prepare the stable-diffusion.cpp backend")

    # List-models command
    list_models_parser = subparsers.add_parser("list-models", help="List all available model aliases")
    list_models_parser.add_argument(
        "--category",
        choices=["text-generation", "coding", "reasoning", "Text + Vision", "text-to-image", "embeddings"],
        help="Filter models by category"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for models by name")
    search_parser.add_argument("query", help="Search term (e.g., 'llama', 'qwen')")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Download a model")
    pull_parser.add_argument("model", help="Model ID, alias, or HF repo")

    # Multimodal commands
    
    # Vision command
    vision_parser = subparsers.add_parser(
        "vision",
        help="Ask questions about images using vision models (qwen-vl, llava, etc.)"
    )
    vision_parser.add_argument("model", help="Vision model alias (e.g., qwen3-vl-7b, llava-v1.6)")
    vision_parser.add_argument("prompt", help="Question or instruction about the image(s)")
    vision_parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Path(s) to image file(s) to analyze"
    )
    vision_parser.add_argument("--max-tokens", type=int, help="Max tokens in response")
    vision_parser.add_argument("--temperature", type=float, help="Sampling temperature")
    vision_parser.add_argument("--no-stream", action="store_true", help="Disable streaming")

    # Image generation command
    genimg_parser = subparsers.add_parser(
        "gen-image",
        help="Generate images from text prompts using stable-diffusion.cpp"
    )
    genimg_parser.add_argument(
        "model",
        help="GGUF model path or HF repo ID containing a GGUF image model - REQUIRED"
    )
    genimg_parser.add_argument(
        "prompt",
        help="Text description of the image to generate"
    )
    genimg_parser.add_argument(
        "--negative",
        help="Negative prompt (what to avoid)"
    )
    genimg_parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)"
    )
    genimg_parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)"
    )
    genimg_parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of sampling steps (default: 28, turbo models use 4)"
    )
    genimg_parser.add_argument(
        "--guidance",
        type=float,
        help="Guidance scale/CFG (default: auto-detect based on model)"
    )
    genimg_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: auto-generated)"
    )
    genimg_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random)"
    )
    genimg_parser.add_argument(
        "--sampler",
        help="Sampling method supported by stable-diffusion.cpp"
    )

    # Video generation command
    genvid_parser = subparsers.add_parser(
        "gen-video",
        help="Deprecated: video generation is not available in the llama.cpp-only build"
    )
    genvid_parser.add_argument("model", help="Video model alias (e.g., wan2.2-5b, mochi-1-10b)")
    genvid_parser.add_argument("prompt", help="Text description of video to generate")
    genvid_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: auto-generated .mp4)"
    )
    genvid_parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames (default: 60)"
    )
    genvid_parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second (default: 24)"
    )
    genvid_parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Video width (default: 512)"
    )
    genvid_parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height (default: 512)"
    )
    genvid_parser.add_argument(
        "--negative",
        help="Negative prompt (what to avoid)"
    )

    # Embed command
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate text embeddings for semantic search and RAG"
    )
    embed_parser.add_argument(
        "model",
        help="Embedding model (e.g., nomic-embed-text, bge-m3, all-minilm-l6-v2)"
    )
    embed_parser.add_argument(
        "prompt",
        nargs="?",
        help='Text to embed (e.g., "Hello world")'
    )
    embed_parser.add_argument(
        "--files",
        nargs="+",
        help="File(s) to process: PDF, DOCX, TXT, JSON (outputs to embeddings.json)"
    )
    embed_parser.add_argument(
        "--batch",
        "-b",
        help="File containing texts to embed (one per line)"
    )
    embed_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file (default: print to stdout, or embeddings.json for --files)"
    )
    embed_parser.add_argument(
        "--format",
        choices=["json", "jsonl", "simple"],
        default="simple",
        help="Output format (default: simple)"
    )
    embed_parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize embeddings"
    )
    embed_parser.add_argument(
        "--no-texts",
        action="store_true",
        help="Don't include original texts in output file"
    )

    # Index command (Knowledge Infrastructure)
    index_parser = subparsers.add_parser(
        "index",
        help="Index files or search local knowledge base"
    )
    index_subparsers = index_parser.add_subparsers(dest="index_command")
    
    # oprel index add <path>
    index_add = index_subparsers.add_parser("add", help="Add files or directory to knowledge store")
    index_add.add_argument("path", help="Path to file or directory")
    
    # oprel index search <query>
    index_search = index_subparsers.add_parser("search", help="Search the local knowledge base")
    index_search.add_argument("query", help="Search query")
    index_search.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    # oprel index sync
    index_subparsers.add_parser("sync", help="Run full sync of all configured sources")

    # Cache commands
    cache_parser = subparsers.add_parser("cache", help="Manage model cache")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command")

    cache_subparsers.add_parser("list", help="List cached models")

    clear_parser = cache_subparsers.add_parser("clear", help="Clear all cached models")
    clear_parser.add_argument("--yes", action="store_true", help="Skip confirmation")

    delete_parser = cache_subparsers.add_parser("delete", help="Delete specific model")
    # Supports alias or filename
    delete_parser.add_argument("model_name", help="Model filename or alias to delete")

    # Parse arguments
    args = parser.parse_args()

    # Set log level
    if args.verbose:
        set_log_level("DEBUG")
    elif args.quiet:
        set_log_level("CRITICAL")

    # Handle run command special case for streaming
    if args.command == "run" and getattr(args, 'no_stream', False):
        args.stream = False


    # Route to command handlers
    if args.command == "chat":
        return cmd_chat(args)
    elif args.command == "recommend":
        return cmd_recommend(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "serve":
        return cmd_serve(args)
    elif args.command == "start":
        return cmd_start(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "models":
        return cmd_models(args)
    elif args.command == "stop":
        return cmd_stop(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "setup":
        if args.setup_command == "runtime":
            return cmd_setup_runtimes(args)
        elif args.setup_command == "image":
            return cmd_setup_image(args)
        else:
            setup_parser.print_help()
            return 1
    elif args.command == "list-models":
        return cmd_list_models(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "pull":
        return cmd_pull(args)
    elif args.command == "vision":
        return cmd_vision(args)
    elif args.command == "gen-image":
        return cmd_gen_image(args)
    elif args.command == "gen-video":
        return _unsupported_feature("Video generation")
    elif args.command == "embed":
        return cmd_embed(args)
    elif args.command == "index":
        if args.index_command == "add":
            return cmd_index(args)
        elif args.index_command == "search":
            return cmd_knowledge_search(args)
        elif args.index_command == "sync":
            return cmd_knowledge_sync(args)
        else:
            index_parser.print_help()
            return 1
    elif args.command == "cache":
        if args.cache_command == "list":
            return cmd_cache_list(args)
        elif args.cache_command == "clear":
            return cmd_cache_clear(args)
        elif args.cache_command == "delete":
            return cmd_cache_delete(args)
        else:
            cache_parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
