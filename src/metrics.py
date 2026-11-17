import numpy as np
import pandas as pd

def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))

def evaluate_retrieval(run: dict, qrels: dict, qids: list, ks=(1, 3, 5, 10)) -> pd.DataFrame:
    rows = []
    for qid in qids:
        rel = qrels[qid]
        good = {c for c, g in rel.items() if g == 2}
        if not good:
            continue
        ranked = run[qid]
        row = {"query_id": qid}
        for k in ks:
            top = set(ranked[:k])
            row[f"Hit@{k}"] = float(bool(good & top))
            row[f"Recall@{k}"] = len(good & top) / len(good)
        row["MRR"] = next((1 / (i + 1) for i, c in enumerate(ranked) if c in good), 0.0)
        gains = [rel.get(c, 0) for c in ranked[:10]]
        ideal = sorted(rel.values(), reverse=True)[:10]
        row["nDCG@10"] = dcg(gains) / dcg(ideal) if dcg(ideal) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
