"""Build a PRACTICE routing table: real weak answers + SIMULATED strong answers.

Only for testing that Phases 3-6 run end to end before the real strong-model run.
Its numbers are meaningless and it is written to data/practice/, never results/.

  python -m src.practice
"""
import os

os.environ["PRACTICE"] = "1"  # must be set before importing config

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.query_models import get_sample, with_prompts  # noqa: E402

# Simulated strong model: usually right when weak is right, often right when weak is wrong.
P_STRONG_IF_WEAK_RIGHT = 0.85
P_STRONG_IF_WEAK_WRONG = 0.45


def main():
    sample = with_prompts(get_sample(config.N_SAMPLES))
    resp = pd.read_json(config.RESPONSES_PATH, lines=True)
    weak = resp[resp["model"] == config.WEAK_MODEL].drop_duplicates("id", keep="last")

    table = sample[["id", "subject", "prompt"]].merge(
        weak[["id", "correct"]].rename(columns={"correct": "weak_correct"}), on="id"
    )
    rng = np.random.default_rng(config.SEED)
    p = np.where(table["weak_correct"], P_STRONG_IF_WEAK_RIGHT, P_STRONG_IF_WEAK_WRONG)
    table["strong_correct"] = rng.random(len(table)) < p

    table.to_csv(config.ROUTING_TABLE_PATH, index=False, encoding="utf-8")
    (config.WORK_DIR / "README.txt").write_text(
        "PRACTICE DATA. strong_correct is SIMULATED, not a real model.\n"
        "Used only to test the code. Do not report any number from this folder.\n",
        encoding="utf-8",
    )
    print(f"PRACTICE table: {len(table)} rows (real weak answers, SIMULATED strong) "
          f"-> {config.ROUTING_TABLE_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
