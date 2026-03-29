"""
AURA-ED Evaluation Script
Computes AUROC, F1, and accuracy for:
  1. Clinical risk scores (MEWS, NEWS, NEWS2, REMS, CART, CCI) — runs immediately
  2. LLM-generated risk tiers — requires GEMINI_API_KEY in .env or Ollama running

Usage:
  python evaluate.py                     # score-based eval only
  python evaluate.py --llm gemini        # score + LLM eval (needs GEMINI_API_KEY)
  python evaluate.py --llm ollama        # score + LLM eval (needs Ollama running)
  python evaluate.py --n 100             # limit LLM eval to N patients
"""

import os, sys, re, argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))

TEST_PATH  = os.path.join(os.path.dirname(__file__), "dataset", "test_master_dataset.csv")
TRAIN_PATH = os.path.join(os.path.dirname(__file__), "dataset", "train_master_dataset.csv")

# Primary outcomes of clinical interest
PRIMARY_OUTCOMES = [
    "outcome_hospitalization",
    "outcome_critical",
    "outcome_icu_transfer_12h",
    "outcome_sepsis",
    "outcome_aki",
    "outcome_acs_mi",
    "outcome_stroke",
    "outcome_ahf",
    "outcome_pneumonia_all",
    "outcome_pe",
]

# Risk score columns and their "high-risk" binary thresholds
SCORE_COLS = {
    "score_MEWS":  3,   # ≥3 = elevated risk (guardrail threshold)
    "score_NEWS":  5,   # ≥5 = elevated risk
    "score_NEWS2": 5,   # ≥5 = escalate care
    "score_REMS":  8,   # ≥8 = elevated risk
    "score_CART":  5,   # ≥5 = elevated risk
    "score_CCI":   2,   # ≥2 = elevated comorbidity burden
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["stay_id"] = df["stay_id"].astype(str)
    return df


def compute_metrics(y_true, y_score_cont, threshold: float) -> dict:
    """Compute AUROC, F1, accuracy from continuous scores and a binary threshold."""
    valid = ~np.isnan(y_score_cont)
    y_true_v = np.array(y_true)[valid]
    y_score_v = np.array(y_score_cont)[valid]
    y_pred_v  = (y_score_v >= threshold).astype(int)

    if len(np.unique(y_true_v)) < 2:
        return {"n": int(valid.sum()), "auroc": None, "f1": None, "accuracy": None,
                "prevalence": float(y_true_v.mean()), "note": "Only one class present"}

    return {
        "n":          int(valid.sum()),
        "prevalence": round(float(y_true_v.mean()), 4),
        "auroc":      round(roc_auc_score(y_true_v, y_score_v), 4),
        "f1":         round(f1_score(y_true_v, y_pred_v, zero_division=0), 4),
        "accuracy":   round(accuracy_score(y_true_v, y_pred_v), 4),
    }


def print_table(title: str, rows: list[dict], columns: list[str]):
    col_w = max(len(c) for c in columns)
    row_w = max(len(r["label"]) for r in rows)
    header = f"{'':>{row_w}}  " + "  ".join(f"{c:>{col_w}}" for c in columns)
    print(f"\n{'─'*len(header)}")
    print(title)
    print('─'*len(header))
    print(header)
    print('─'*len(header))
    for r in rows:
        vals = "  ".join(
            f"{r.get(c, '—'):>{col_w}}" if isinstance(r.get(c), str)
            else f"{r.get(c, float('nan')):>{col_w}.4f}" if r.get(c) is not None
            else f"{'—':>{col_w}}"
            for c in columns
        )
        print(f"{r['label']:>{row_w}}  {vals}")
    print('─'*len(header))


def eval_scores(df: pd.DataFrame, split_name: str):
    print(f"\n{'='*70}")
    print(f"SCORE-BASED EVALUATION  —  {split_name}  ({len(df):,} patients)")
    print(f"{'='*70}")

    for score_col, threshold in SCORE_COLS.items():
        if score_col not in df.columns:
            print(f"\n[skip] {score_col} not in dataset")
            continue

        rows = []
        for outcome in PRIMARY_OUTCOMES:
            if outcome not in df.columns:
                continue
            y_true = df[outcome].fillna(0).astype(int)
            y_score = df[score_col]
            m = compute_metrics(y_true, y_score.values, threshold)
            rows.append({
                "label":      outcome.replace("outcome_", ""),
                "n":          str(m["n"]),
                "prevalence": m["prevalence"],
                "auroc":      m["auroc"],
                "f1":         m["f1"],
                "accuracy":   m["accuracy"],
            })

        print_table(
            f"{score_col}  (binary threshold ≥ {threshold})",
            rows,
            ["auroc", "f1", "accuracy", "prevalence"],
        )


RISK_TIER_RE = re.compile(
    r"##\s*Overall Risk Assessment.*?(?:LOW|MODERATE|HIGH|CRITICAL)", re.IGNORECASE | re.DOTALL
)

def parse_risk_tier(brief: str) -> str | None:
    """Extract the risk tier keyword from an AURA brief."""
    match = re.search(
        r"##\s*Overall Risk Assessment[^\n]*\n[^\n]*\b(CRITICAL|HIGH|MODERATE|LOW)\b",
        brief, re.IGNORECASE
    )
    if not match:
        match = re.search(r"\b(CRITICAL|HIGH|MODERATE|LOW)\b", brief, re.IGNORECASE)
    return match.group(1).upper() if match else None


TIER_SCORE = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


def generate_brief_llm(summary: dict, provider: str, model: str,
                        ollama_client=None, gemini_client=None) -> str:
    from AURA_ED_eval_helpers import build_prompt  
    prompt = build_prompt(summary)
    if provider == "gemini":
        resp = gemini_client.models.generate_content(model=model, contents=prompt)
        return resp.text
    else:
        resp = ollama_client.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return resp.message.content


def eval_llm(df: pd.DataFrame, provider: str, model: str, n: int, split_name: str):
    """Run the LLM on n test patients and compute metrics from parsed risk tiers."""
    from dotenv import load_dotenv
    load_dotenv()

    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "aura_app", os.path.join(os.path.dirname(__file__), "AURA-ED.py")
    )
    import unittest.mock as mock
    with mock.patch.dict(sys.modules, {"streamlit": mock.MagicMock()}):
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)

    extract_patient_summary = app.extract_patient_summary
    build_prompt = app.build_prompt

    if provider == "gemini":
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            print("\n[ERROR] GEMINI_API_KEY not set in .env — skipping LLM eval")
            return
        client = genai.Client(api_key=api_key)
        def call_llm(prompt):
            return client.models.generate_content(model=model, contents=prompt).text
    else:
        import ollama as ollama_lib
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama_lib.Client(host=host)
        def call_llm(prompt):
            return client.chat(model=model,
                               messages=[{"role": "user", "content": prompt}]).message.content

    sample = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    print(f"\n{'='*70}")
    print(f"LLM-BASED EVALUATION  —  {split_name}  ({provider}/{model}, n={len(sample)})")
    print(f"{'='*70}")

    tiers, true_labels = [], {o: [] for o in PRIMARY_OUTCOMES}
    errors = 0

    for i, (_, row) in enumerate(sample.iterrows()):
        try:
            summary = extract_patient_summary(row)
            prompt  = build_prompt(summary)
            brief   = call_llm(prompt)
            tier    = parse_risk_tier(brief)
            tiers.append(TIER_SCORE.get(tier, np.nan) if tier else np.nan)
            for o in PRIMARY_OUTCOMES:
                if o in row:
                    v = row[o]
                    true_labels[o].append(int(v) if pd.notna(v) else 0)
                else:
                    true_labels[o].append(0)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(sample)} completed …")
        except Exception as e:
            tiers.append(np.nan)
            for o in PRIMARY_OUTCOMES:
                true_labels[o].append(0)
            errors += 1

    if errors:
        print(f"  [{errors} errors during generation]")

    y_score = np.array(tiers, dtype=float)
    rows = []
    for outcome in PRIMARY_OUTCOMES:
        if outcome not in df.columns:
            continue
        y_true = np.array(true_labels[outcome])
        m = compute_metrics(y_true, y_score, threshold=2.0)  # when ≥2 = it would be consideredHIGH/CRITICAL
        rows.append({
            "label":      outcome.replace("outcome_", ""),
            "n":          str(m["n"]),
            "prevalence": m["prevalence"],
            "auroc":      m["auroc"],
            "f1":         m["f1"],
            "accuracy":   m["accuracy"],
        })

    print_table(
        f"LLM risk tier  (HIGH/CRITICAL = positive, threshold ≥ 2)",
        rows,
        ["auroc", "f1", "accuracy", "prevalence"],
    )


def main():
    parser = argparse.ArgumentParser(description="AURA-ED evaluation")
    parser.add_argument("--llm",   choices=["gemini", "ollama"], default=None,
                        help="Enable LLM-based evaluation")
    parser.add_argument("--model", default=None,
                        help="Model name (default: gemini-2.0-flash / llama3.2)")
    parser.add_argument("--n",     type=int, default=100,
                        help="Number of test patients for LLM eval (default: 100)")
    parser.add_argument("--split", choices=["test", "train", "both"], default="test",
                        help="Which split to evaluate (default: test)")
    args = parser.parse_args()

    splits = []
    if args.split in ("test", "both"):
        splits.append((TEST_PATH, "TEST SET"))
    if args.split in ("train", "both"):
        splits.append((TRAIN_PATH, "TRAIN SET"))

    for path, name in splits:
        print(f"\nLoading {name} from {path} …")
        df = load_data(path)
        eval_scores(df, name)

        if args.llm:
            default_model = "gemini-2.0-flash" if args.llm == "gemini" else "llama3.2"
            model = args.model or default_model
            eval_llm(df, args.llm, model, args.n, name)

    print("\nDone.")


if __name__ == "__main__":
    main()
