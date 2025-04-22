# AudioGist Logging System Changes

## Original Implementation
The original logging system in AudioGist created separate log files for each task:
- Each task (transcription, summarization, etc.) generated its own log file
- Log files were named with pattern: `audiogist-{task_name}-{timestamp}.log`
- This resulted in multiple log files for a single application run

## New Implementation
The updated logging system uses a session-based approach:
- A single log file is created for the entire application session
- All tasks log to the same file during a single application run
- Log files are named with pattern: `audiogist-{YYYYMMDD-HHMM}.log`
- The session ID is created once when the application starts
- Task names are preserved in the log entries as logger names

## Key Changes Made

1. Added a global `SESSION_ID` variable that's created once at application startup:
   ```python
   SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M")
   ```

2. Added a global `SESSION_LOG_PATH` variable to track the log file path

3. Modified the `setup_logging()` function to:
   - Create the log file only on first call
   - Use the session ID in the filename
   - Return a named logger for each task instead of reconfiguring the root logger

4. Updated the log format to include the logger name (task name):
   ```python
   format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
   ```

5. Updated the README.md documentation to reflect the new logging behavior

## Benefits
- Easier to follow the complete flow of operations in a single log file
- Reduced file system clutter
- Maintains task context through logger names
- Session ID provides clear separation between application runs
