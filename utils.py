from datetime import datetime

def log_flush(fileIO, txt: str):
    """
    Write and flush to the disk.
    """
    fileIO.write(txt)
    fileIO.write("\n")
    fileIO.flush()

def get_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def get_timestamp_ms() -> str:
    return datetime.now().strftime('%m-%d_%H:%M:%S.%f')
