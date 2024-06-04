from datetime import datetime

TRAIN_EPOCHS = 15
BATCH_SIZE = 64
SAVED_MODELS_DIR = "saved_models"
LOGS_DIR = "logs"

def get_timestamp() -> str:
   return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_saved_model_path(model_name: str, timestamp: str) -> str:
    return f"{SAVED_MODELS_DIR}/{model_name}-{timestamp}.pt"

def get_log_dir(model_name: str, timestamp: str) -> str:
    return f"{LOGS_DIR}/{model_name}-{timestamp}/"