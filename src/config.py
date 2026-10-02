"""Central configuration. Change values here, not inside the other scripts."""
import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# PRACTICE mode (set env var PRACTICE=1): the strong-model answers are SIMULATED, only to
# test that the code runs. Everything is then read from / written to data/practice/,
# never results/, so fake numbers can't mix with real ones.
PRACTICE = os.getenv("PRACTICE") == "1"
WORK_DIR = DATA_DIR / "practice" if PRACTICE else DATA_DIR
RESULTS_DIR = WORK_DIR / "results" if PRACTICE else ROOT / "results"
WORK_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESPONSES_PATH = DATA_DIR / "responses.jsonl"  # always the real model answers
ROUTING_TABLE_PATH = WORK_DIR / "routing_table.csv"
SPLITS_PATH = WORK_DIR / "splits.json"
EMBED_CACHE_PATH = DATA_DIR / "embeddings_e5_small.npz"  # {ids, vectors}, keyed by question id

# --- Secrets ---
load_dotenv(ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Data ---
DATASET_NAME = "MBZUAI/ArabicMMLU"
N_SAMPLES = 1500  # pilot was N=10
SEED = 42

# --- Models ---
# Both models run locally on CPU ($0). HF free API credits ran out during the pilot,
# where the pair was Qwen2.5-1.5B (weak) vs Qwen2.5-72B via API (strong).
WEAK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
STRONG_MODEL = "Qwen/Qwen2.5-3B-Instruct"
STRONG_BACKEND = "local"  # "local" (transformers on CPU) or "api" (HF Inference Providers)
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.0

# --- Routers ---
EMBED_MODEL = "intfloat/multilingual-e5-small"
TEST_SIZE = 0.30
N_GROUP_FOLDS = 5

# --- Label sanity warnings (Phase 3) ---
WEAK_TOO_GOOD = 0.90       # weak accuracy above this -> little room for routing to help
NEAR_CHANCE_MARGIN = 0.05  # weak accuracy within 5 points of random guessing -> warn

# --- Evaluation ---
STRONG_PCTS = list(range(0, 101, 5))  # % of questions routed to the strong model
N_BOOTSTRAP = 1000
