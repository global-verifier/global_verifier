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

def is_success_trail(score):
    return score > 0

def extract_exp_ids(experiences: list) -> list:
    return [exp['id'] for exp in experiences]
