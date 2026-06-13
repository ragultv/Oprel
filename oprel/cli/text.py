"""
Text generation and chat commands for Oprel CLI.
"""
import argparse
import sys
import uuid
from oprel import Model, __version__
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive chat mode"""
    from ..utils.chat_templates import format_chat_prompt
    print(f"Oprel Chat v{__version__}")
    print(f"Model: {args.model}")
    print("Type 'exit', 'quit' or Ctrl+D to end.")
    print("Type '/reset' to clear conversation history.\n")

    # Determine server mode
    use_server = not getattr(args, 'no_server', False)
    
    # Generate conversation ID for tracking history on server
    conversation_id = str(uuid.uuid4())
    if use_server:
        print(f"Conversation ID: {conversation_id}")
        
    system_prompt = getattr(args, 'system', None)
    if system_prompt:
        print(f"System: {system_prompt}")

    try:
        with Model(
            args.model,
            quantization=args.quantization,
            max_memory_mb=args.max_memory,
            use_server=use_server,
            allow_low_quality=getattr(args, 'allow_low_quality', False),
        ) as model:
            print("\nModel loaded. Ready to chat!\n")

            # Interactive loop across platforms
            import sys

            # Local conversation history for non-server (direct) mode
            conversation_history = []

            while True:
                try:
                    # Handle input properly (Python input() uses readline if available)
                    try:
                        prompt = input(">>> ")
                    except EOFError:
                        print("\nExiting...")
                        break

                    if prompt.lower() in ["exit", "quit"]:
                        break

                    if prompt.strip() == "/reset":
                        if use_server:
                            # Generate new conversation ID to reset history
                            conversation_id = str(uuid.uuid4())
                            print(f"Conversation reset. New ID: {conversation_id}\n")
                        else:
                            conversation_history = []
                            print("Conversation history cleared (local mode).\n")
                        continue

                    if not prompt.strip():
                        continue

                    # Prepare formatted prompt using chat templates for direct mode
                    if use_server:
                        formatted_prompt = prompt
                    else:
                        formatted_prompt = format_chat_prompt(
                            model_id=args.model,
                            user_message=prompt,
                            system_prompt=system_prompt,
                            conversation_history=conversation_history,
                        )

                    # Server mode or direct mode both support streaming
                    if args.stream:
                        assistant_accum = ""
                        print("AI: ", end="", flush=True)
                        for token in model.generate(
                            formatted_prompt,
                            stream=True,
                            conversation_id=conversation_id if use_server else None,
                            system_prompt=system_prompt if use_server else None,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            thinking=getattr(args, 'thinking', False),
                            rag=getattr(args, 'rag', False),
                        ):
                            print(token, end="", flush=True)
                            assistant_accum += token
                        print()

                        # Update conversation history in local mode
                        if not use_server:
                            conversation_history.append({"role": "user", "content": prompt})
                            conversation_history.append({"role": "assistant", "content": assistant_accum})
                            if len(conversation_history) > 40:
                                conversation_history = conversation_history[-40:]

                    else:
                        response = model.generate(
                            formatted_prompt,
                            conversation_id=conversation_id if use_server else None,
                            system_prompt=system_prompt if use_server else None,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            thinking=getattr(args, 'thinking', False),
                            rag=getattr(args, 'rag', False),
                        )
                        print(response)
                        print()

                        if not use_server:
                            conversation_history.append({"role": "user", "content": prompt})
                            conversation_history.append({"role": "assistant", "content": response})
                            if len(conversation_history) > 40:
                                conversation_history = conversation_history[-40:]

                    # Clear one-time system prompt after first use
                    system_prompt = None

                except KeyboardInterrupt:
                    print("\nInterrupted. Type 'exit' to quit.")
                    continue

        return 0

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return 1


def cmd_generate(args: argparse.Namespace) -> int:
    """Single-shot text generation"""
    from ..utils.chat_templates import format_chat_prompt
    
    # Determine server mode
    use_server = not getattr(args, 'no_server', False)

    try:
        with Model(
            args.model,
            quantization=args.quantization,
            max_memory_mb=args.max_memory,
            use_server=use_server,
            allow_low_quality=getattr(args, 'allow_low_quality', False),
        ) as model:
            # Format prompt using chat templates
            formatted_prompt = format_chat_prompt(
                model_id=args.model,
                user_message=args.prompt,
                system_prompt=None,
                conversation_history=[]
            )
            
            response = model.generate(
                formatted_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=args.stream,
                thinking=getattr(args, 'thinking', False),
                rag=getattr(args, 'rag', False),
            )

            if args.stream:
                for token in response:
                    print(token, end="", flush=True)
                print()
            else:
                print(response)

        return 0

    except Exception as e:
        logger.error(f"Generate error: {e}")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Fast inference using direct mode (auto-cleanup after use)"""
    import sys
    from ..utils.chat_templates import format_chat_prompt
    
    try:
        # Prefer using the daemon server if it's running, unless the user
        # explicitly passed --no-server. This avoids spawning a second
        # heavyweight model process and consuming duplicate memory.
        use_server = not getattr(args, 'no_server', False)
        model = Model(
            args.model,
            quantization=args.quantization,
            use_server=use_server,
            allow_low_quality=getattr(args, 'allow_low_quality', False),
        )
        
        # Load model
        model.load()
        
        # Flush stderr to ensure all log messages are written before output
        sys.stderr.flush()
        
        # Add separator between logs and response
        print()
        
        # If no prompt provided, enter interactive mode
        if args.prompt is None:
            result = _run_interactive(model, args)
            # Cleanup after interactive session
            model.unload()
            return result
        
        # One-shot mode: generate single response with proper chat formatting
        formatted_prompt = format_chat_prompt(
            model_id=args.model,
            user_message=args.prompt,
            system_prompt=None,
            conversation_history=[]
        )
        
        if args.stream:
            for token in model.generate(
                formatted_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=True,
                thinking=getattr(args, 'thinking', False),
                rag=getattr(args, 'rag', False),
            ):
                print(token, end="", flush=True)
            print()
        else:
            response = model.generate(
                formatted_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                thinking=getattr(args, 'thinking', False),
                rag=getattr(args, 'rag', False),
            )
            print(response)
        
        # CRITICAL: Unload model to free GPU/RAM
        # This stops the backend process and releases all memory
        logger.debug("Cleaning up: Unloading model to free GPU/RAM...")
        model.unload()
        logger.debug("✓ Memory freed successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"Run error: {e}")
        # Ensure cleanup even on error
        try:
            if 'model' in locals():
                model.unload()
        except:
            pass
        return 1


def _run_interactive(model: Model, args: argparse.Namespace) -> int:
    """Interactive chat mode for oprel run (like ollama run)"""
    import sys
    import time
    import threading
    from ..utils.chat_templates import format_chat_prompt
    
    print(f">>> Model loaded: {args.model}")
    print(">>> Send a message (/? for help)")
    print(">>> Auto-cleanup: Model will unload after 15 minutes of inactivity")
    print()
    
    # Generate conversation ID for tracking history on server
    conversation_id = str(uuid.uuid4())
    system_prompt = getattr(args, 'system', None)

    # Maintain a local conversation history (for direct interactive mode)
    # Each entry: {'role': 'user'|'assistant', 'content': '...'}
    conversation_history = []
    
    # Idle timeout tracking (15 minutes)
    IDLE_TIMEOUT_SECONDS = 15 * 60
    last_activity_time = time.time()
    idle_check_lock = threading.Lock()
    should_exit = threading.Event()
    
    def idle_monitor():
        """Background thread to monitor idle time and trigger cleanup"""
        nonlocal last_activity_time
        while not should_exit.is_set():
            time.sleep(60)  # Check every minute
            
            with idle_check_lock:
                idle_time = time.time() - last_activity_time
                
            if idle_time > IDLE_TIMEOUT_SECONDS:
                print("\n\n⏱️  Idle timeout reached (15 minutes)")
                print("Unloading model to free memory...")
                print("Bye!\n")
                should_exit.set()
                # Force exit the input loop
                import os
                if hasattr(sys, 'platform') and sys.platform == "win32":
                    try:
                        import msvcrt
                        # Send Ctrl+C to interrupt input() on Windows
                        os.kill(os.getpid(), 2)  # SIGINT
                    except:
                        pass
                return
    
    # Start idle monitor thread
    monitor_thread = threading.Thread(target=idle_monitor, daemon=True)
    monitor_thread.start()
    
    try:
        while not should_exit.is_set():
            try:
                # Get user input
                try:
                    user_input = input(">>> ")
                    
                    # Update activity timestamp
                    with idle_check_lock:
                        last_activity_time = time.time()
                        
                except EOFError:
                    print("\nBye!")
                    break
                
                # Check if we should exit due to idle timeout
                if should_exit.is_set():
                    break
                
                # Handle special commands
                if user_input.strip() in ["/exit", "/bye", "/quit"]:
                    print("Bye!")
                    break
                
                if user_input.strip() == "/?":
                    print("Available commands:")
                    print("  /exit, /bye, /quit - Exit the chat")
                    print("  /reset            - Clear conversation history")
                    print("  /?                - Show this help")
                    print()
                    continue
                
                if user_input.strip() == "/reset":
                    conversation_id = str(uuid.uuid4())
                    conversation_history.clear()
                    print("Conversation history cleared.\n")
                    continue
                
                if not user_input.strip():
                    continue
                
                # Generate response - use chat templates for consistent formatting
                try:
                    # Format prompt using chat templates so models receive proper roles
                    formatted_prompt = format_chat_prompt(
                        model_id=args.model,
                        user_message=user_input,
                        system_prompt=system_prompt,
                        conversation_history=conversation_history,
                    )

                    if args.stream:
                        # Stream tokens and accumulate assistant response
                        assistant_accum = ""
                        print("AI: ", end="", flush=True)
                        for token in model.generate(
                            formatted_prompt,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            stream=True,
                            conversation_id=conversation_id,
                            system_prompt=None,
                            thinking=getattr(args, 'thinking', False),
                            rag=getattr(args, 'rag', False),
                        ):
                            print(token, end="", flush=True)
                            assistant_accum += token
                        print("\n")
                        # Append user and assistant to local history
                        conversation_history.append({"role": "user", "content": user_input})
                        conversation_history.append({"role": "assistant", "content": assistant_accum})
                        # Truncate history to recent 20 turns to limit prompt size
                        if len(conversation_history) > 40:
                            conversation_history = conversation_history[-40:]
                        system_prompt = None

                    else:
                        response = model.generate(
                            formatted_prompt,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            conversation_id=conversation_id,
                            system_prompt=None,
                            thinking=getattr(args, 'thinking', False),
                            rag=getattr(args, 'rag', False),
                        )
                        # Append to history and display
                        conversation_history.append({"role": "user", "content": user_input})
                        conversation_history.append({"role": "assistant", "content": response})
                        if len(conversation_history) > 40:
                            conversation_history = conversation_history[-40:]
                        system_prompt = None
                        print(response)
                        print()
                    
                    # Update activity timestamp after successful generation
                    with idle_check_lock:
                        last_activity_time = time.time()
                
                except KeyboardInterrupt:
                    print("\n")
                    continue
                except Exception as e:
                    print(f"\nError: {e}\n")
                    continue
            
            except KeyboardInterrupt:
                print("\n\nUse /exit to quit\n")
                continue
    
    finally:
        # Signal monitor thread to stop
        should_exit.set()
        
        # Explicitly unload the model and force-stop backend to free GPU memory
        try:
            print("\nCleaning up...")
            
            # Force-stop backend process (critical for GPU memory cleanup)
            if hasattr(model, '_process') and model._process:
                print("Stopping backend process...")
                model._process.stop(force=False)
                model._process = None
            
            # Stop monitor if exists
            if hasattr(model, '_monitor') and model._monitor:
                model._monitor.stop()
                model._monitor = None
            
            # Mark as unloaded
            model._loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("✓ GPU memory cleared")
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Could not clear CUDA cache: {e}")
            
            print("✓ Cleanup complete")
                
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
    
    return 0
