"""
Terminal Log Capture System
Captures all stdout, stderr, and logging output to timestamped .txt files
"""
import sys
import os
from datetime import datetime
from typing import TextIO
import io
import atexit

# Global variables to track the log file
_terminal_log_file: TextIO = None
_terminal_log_path: str = None
_original_stdout: TextIO = None
_original_stderr: TextIO = None
_is_setup: bool = False
_main_session_start_time: str = None  # Timestamp of the main session start


class TeeOutput:
    """A class that writes to both a file and the original stdout/stderr"""
    
    def __init__(self, original_stream: TextIO, log_file: TextIO):
        self.original_stream = original_stream
        self.log_file = log_file
        self.encoding = getattr(original_stream, 'encoding', 'utf-8')
        self.errors = getattr(original_stream, 'errors', 'replace')
        # Buffer for incomplete lines (in case of partial writes)
        self.buffer = ''
    
    def write(self, text: str) -> int:
        """Write to both the original stream and the log file"""
        if not text:
            return 0
        
        # Handle encoding issues
        if isinstance(text, bytes):
            try:
                text = text.decode(self.encoding, errors=self.errors)
            except (UnicodeDecodeError, AttributeError):
                text = text.decode('utf-8', errors='replace')
        
        # Add to buffer (for handling partial lines)
        self.buffer += text
        
        # Write to original stream (terminal) first
        try:
            if self.original_stream:
                self.original_stream.write(text)
                # Flush immediately to ensure real-time output
                self.original_stream.flush()
        except (OSError, ValueError, AttributeError):
            pass  # Terminal might be closed
        
        # Write to log file
        try:
            if self.log_file:
                self.log_file.write(text)
                # Flush immediately to ensure everything is written
                self.log_file.flush()
        except (OSError, ValueError, UnicodeEncodeError, AttributeError) as e:
            # If file writing fails, try to write error to original stream
            try:
                if self.original_stream:
                    error_msg = f"[LOG ERROR] Failed to write to log file: {e}\n"
                    self.original_stream.write(error_msg)
                    self.original_stream.flush()
            except:
                pass
        
        return len(text)
    
    def flush(self):
        """Flush both streams"""
        try:
            if self.original_stream:
                self.original_stream.flush()
        except (OSError, ValueError):
            pass
        try:
            if self.log_file:
                self.log_file.flush()
        except (OSError, ValueError):
            pass
    
    def close(self):
        """Close the log file (but keep original stream open)"""
        if self.log_file:
            try:
                self.log_file.close()
            except:
                pass


def setup_terminal_logging(log_dir: str = 'data/terminal_logs', reuse_existing: bool = True, redirect_streams: bool = False):
    """
    Set up terminal logging to capture logging output to timestamped .txt files.
    
    Args:
        log_dir: Directory to store terminal log files (default: 'data/terminal_logs')
        reuse_existing: If True and a log file already exists, reuse it instead of creating a new one
        redirect_streams: If True, redirect stdout/stderr to capture print() statements. 
                         If False (default), only capture logging output (terminal works normally)
    
    Returns:
        str: Path to the created log file
    """
    global _terminal_log_file, _terminal_log_path, _original_stdout, _original_stderr, _is_setup, _main_session_start_time
    
    # Flask's reloader creates a new process each time it reloads
    # Check if we're in the reloader process (child process)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    
    # If logging is already set up in THIS process, reuse it
    if reuse_existing and _is_setup and _terminal_log_path:
        try:
            # Verify file is still accessible
            if os.path.exists(_terminal_log_path):
                if _terminal_log_file:
                    _terminal_log_file.flush()
                return _terminal_log_path
        except (OSError, ValueError, AttributeError):
            # File might be closed, continue to create new one
            pass
    
    try:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate timestamped filename
        # For the main process, use the session start time
        # For reloader processes, use the same session time but mark as reload
        if not is_reloader_process:
            # Main process - set the session start time
            _main_session_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_filename = f'terminal_log_{_main_session_start_time}.txt'
        else:
            # Reloader process - use the main session time if available, otherwise current time
            session_time = _main_session_start_time or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_filename = f'terminal_log_{session_time}_reload_pid{os.getpid()}.txt'
        
        _terminal_log_path = os.path.join(log_dir, log_filename)
        
        # Open log file with UTF-8 encoding
        _terminal_log_file = open(_terminal_log_path, 'w', encoding='utf-8', errors='replace')
        
        # Write header to log file (only for main process, not reloader)
        if not is_reloader_process:
            header = f"""
{'='*80}
MOTHER AI - Terminal Log
Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log File: {_terminal_log_path}
Process ID: {os.getpid()}
{'='*80}

"""
            _terminal_log_file.write(header)
            _terminal_log_file.flush()
        else:
            # For reloader process, add a separator
            separator = f"\n{'='*80}\n[FLASK RELOAD] Process restarted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (PID: {os.getpid()})\n{'='*80}\n\n"
            _terminal_log_file.write(separator)
            _terminal_log_file.flush()
        
        import logging
        
        # Get root logger
        root_logger = logging.getLogger()
        
        # Add a file handler to capture all logging to the .txt file
        # This captures all logging output without interfering with terminal display
        file_handler = logging.FileHandler(_terminal_log_path, mode='a', encoding='utf-8', errors='replace')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.DEBUG)
        
        # Only add if not already present
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(_terminal_log_path)
            for h in root_logger.handlers
        ):
            root_logger.addHandler(file_handler)
        
        # IMPORTANT: Ensure there's a console handler for terminal output
        # Check if there's already a console handler
        has_console_handler = any(
            isinstance(h, logging.StreamHandler) and h.stream in (sys.stdout, sys.stderr)
            for h in root_logger.handlers
        )
        
        if not has_console_handler:
            # Add console handler to show logs in terminal
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            console_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(console_handler)
        
        # Ensure root logger level is set to DEBUG so all logs are shown
        root_logger.setLevel(logging.DEBUG)
        
        # If redirect_streams is True, also redirect stdout/stderr to capture print() statements
        if redirect_streams:
            # Save original stdout and stderr
            _original_stdout = sys.stdout
            _original_stderr = sys.stderr
            
            # Replace stdout and stderr with TeeOutput
            sys.stdout = TeeOutput(_original_stdout, _terminal_log_file)
            sys.stderr = TeeOutput(_original_stderr, _terminal_log_file)
            
            # Update existing handlers to use redirected streams
            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    if handler.stream == _original_stdout:
                        handler.stream = sys.stdout
                    elif handler.stream == _original_stderr:
                        handler.stream = sys.stderr
        else:
            # Don't redirect streams - terminal works normally
            # Only file handler will capture logging
            _original_stdout = sys.stdout
            _original_stderr = sys.stderr
        
        # Mark as set up
        _is_setup = True
        
        # Print confirmation (will appear in terminal and file if redirect_streams=True, or just terminal if False)
        print(f"[TERMINAL LOG] Logging output is being saved to: {_terminal_log_path}")
        if not redirect_streams:
            print(f"[TERMINAL LOG] Terminal output is normal. Only logging is captured to file.")
        
        return _terminal_log_path
        
    except Exception as e:
        # If setup fails, try to write error to stderr
        try:
            sys.stderr.write(f"[TERMINAL LOG ERROR] Failed to setup terminal logging: {e}\n")
        except:
            pass
        return None


def close_terminal_logging():
    """Close the terminal log file and restore original stdout/stderr"""
    global _terminal_log_file, _terminal_log_path, _original_stdout, _original_stderr
    
    try:
        if _terminal_log_file:
            # Write footer
            footer = f"""
{'='*80}
Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
            _terminal_log_file.write(footer)
            _terminal_log_file.flush()
            _terminal_log_file.close()
            _terminal_log_file = None
        
        # Restore original stdout and stderr
        if _original_stdout:
            sys.stdout = _original_stdout
        if _original_stderr:
            sys.stderr = _original_stderr
        
        print(f"[TERMINAL LOG] Terminal logging closed. Log saved to: {_terminal_log_path}")
        _terminal_log_path = None
        
    except Exception as e:
        try:
            sys.stderr.write(f"[TERMINAL LOG ERROR] Failed to close terminal logging: {e}\n")
        except:
            pass


def get_terminal_log_path() -> str:
    """Get the path to the current terminal log file"""
    return _terminal_log_path


# Register cleanup on exit
atexit.register(close_terminal_logging)

