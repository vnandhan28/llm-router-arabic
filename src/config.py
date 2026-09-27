"""Central configuration. Change values here, not inside the other scripts."""
import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RESPONSES_PATH = DATA_DIR / "responses.jsonl"
ROUTING_TABLE_PATH = DATA_DIR / "routing_table.csv"
EMBED_CACHE_PATH = DATA_DIR / "embeddings_e5_small.npy"

# --- Secrets ---
load_dotenv(ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Data ---
DATASET_NAME = "MBZUAI/ArabicMMLU"
N_SAMPLES = 20  # start small; raise to 1500 after the pilot run
SEED = 42

# --- Models ---
WEAK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # local, CPU
WEAK_MODEL_FALLBACK = "Qwen/Qwen2.5-0.5B-Instruct"
STRONG_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # via HF Inference Providers; confirmed in Phase 2
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.0

# --- Routers ---
EMBED_MODEL = "intfloat/multilingual-e5-small"
TEST_SIZE = 0.30
N_GROUP_FOLDS = 5

# --- Evaluation ---
STRONG_PCTS = list(range(0, 101, 5))  # % of questions routed to the strong model
N_BOOTSTRAP = 1000
