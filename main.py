# main.py
import sys
import os

# Setup terminal logging FIRST, before any other imports that might print
# Set redirect_streams=True to capture all terminal output (print statements, stdout, stderr) to log files
# Output will still appear in terminal AND be logged to file
try:
    from utils.terminal_logger import setup_terminal_logging
    log_path = setup_terminal_logging(log_dir='data/terminal_logs', redirect_streams=True)
    if log_path:
        print(f"[INIT] Terminal logging initialized: {log_path}")
        # Force flush to ensure this message is in the log
        sys.stdout.flush()
except Exception as e:
    # If terminal logging fails, continue anyway
    print(f"[WARNING] Terminal logging setup failed: {e}", file=sys.stderr)
    sys.stderr.flush()

# Configure Werkzeug/Flask logging to use our redirected streams
import logging
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.INFO)
# Remove any existing handlers
werkzeug_logger.handlers = []
# Add handler that uses our redirected stdout
werkzeug_handler = logging.StreamHandler(sys.stdout)
werkzeug_handler.setFormatter(logging.Formatter('%(message)s'))
werkzeug_logger.addHandler(werkzeug_handler)

from app import create_app # Import the application factory function

if __name__ == '__main__':
    app = create_app() # Call the factory function to create the app instance
    try:
        # Ensure all output is flushed before starting Flask
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Disable Flask's reloader to prevent multiple log files
        # The reloader creates a new process which creates a new log file each time
        # If you need auto-reload, you can enable it, but it will create separate log files for each reload
        # For a single log file per session, disable the reloader
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Received interrupt signal, shutting down...")
        sys.stdout.flush()
        from utils.terminal_logger import close_terminal_logging
        close_terminal_logging()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Application error: {e}")
        sys.stdout.flush()
        sys.stderr.flush()
        from utils.terminal_logger import close_terminal_logging
        close_terminal_logging()
        raise
