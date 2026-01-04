from pathlib import Path
import json
import re
import pandas as pd
import pyterrier as pt
import lightgbm as lgb
import os

# === Pfade ===
INDEX_DIR = Path("var/src/pt_index")  # <-- passt jetzt zu deinem existierenden Index
QUERIES_PATH = Path("data/queries.txt")
QRELS_PATH = Path("data/qrels.jsonl")

RUN_BM25_TEST = Path("runs/pt_bm25_test.jsonl")
RUN_LTR_TEST  = Path("runs/pt_ltr_test.jsonl")

# === Parameter ===
CAND_TOPK = 1000
TRAIN_TOPK = 500
TEST_FRAC = 0.2
SEED = 42
MODEL_NUM_LEAVES = 63
MODEL_NUM_TREES = 400
MODEL_LEARNING_RATE = 0.05

# === Hilfsfunktionen ===
def read_queries_tsv(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                qid, query = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise ValueError(f"Bad query line: {line}")
                qid, query = parts
            rows.append({"qid": str(qid).strip(), "query": str(query).strip()})
    return pd.DataFrame(rows)

def read_qrels_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({"qid": str(obj["qid"]), "docno": str(obj["doc_id"]), "label": int(obj["rel"])})
    return pd.DataFrame(rows)

def qlen(text: str) -> int:
    return len(re.findall(r"\w+", text.lower()))

def ensure_java():
    if not pt.java.started():
        pt.java.init()

def write_run_jsonl(df: pd.DataFrame, out_path: Path, score_col: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)  # Ordner automatisch erstellen
    with out_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            f.write(json.dumps({
                "qid": str(row.qid),
                "doc_id": str(row.docno),
                "rank": int(row.new_rank),
                "score": float(getattr(row, score_col)),
            }) + "\n")

# === Main ===
def main():
    ensure_java()

    # Prüfe Index-Pfad
    index_path = INDEX_DIR.resolve()
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    # Index laden
    idx = pt.IndexRef.of(str(index_path))
    bm25 = pt.terrier.Retriever(
        idx,
        wmodel="BM25",
        num_results=CAND_TOPK,
        metadata=["docno", "doclen", "title_len", "abs_len", "year"]
    )

    # Queries & Qrels laden
    queries = read_queries_tsv(QUERIES_PATH.resolve())
    qrels = read_qrels_jsonl(QRELS_PATH.resolve())

    # Train/Test split
    qids = queries["qid"].tolist()
    qids_shuffled = pd.Series(qids).sample(frac=1.0, random_state=SEED).tolist()
    cut = int(len(qids_shuffled) * (1 - TEST_FRAC))
    train_qids = set(qids_shuffled[:cut])
    test_qids = set(qids_shuffled[cut:])
    q_train = queries[queries["qid"].isin(train_qids)].reset_index(drop=True)
    q_test  = queries[queries["qid"].isin(test_qids)].reset_index(drop=True)
    print(f"SPLIT: train={len(q_train)} | test={len(q_test)}")

    # BM25 candidates
    print("BM25: candidates train...")
    cand_train = bm25.transform(q_train)
    print("BM25: candidates test...")
    cand_test = bm25.transform(q_test)

    # Feature preparation
    qlen_map_train = dict(zip(q_train["qid"], q_train["query"].map(qlen)))
    qlen_map_test  = dict(zip(q_test["qid"], q_test["query"].map(qlen)))

    def prep(df: pd.DataFrame, qlen_map: dict) -> pd.DataFrame:
        out = df.copy()
        out["f_bm25"] = out["score"].astype(float)
        out["f_qlen"] = out["qid"].map(qlen_map).fillna(0).astype(int)
        for col in ["doclen", "title_len", "abs_len", "year"]:
            out[col] = pd.to_numeric(out[col].fillna(0), errors="coerce").astype(int)
        out["f_doclen"] = out["doclen"]
        out["f_title_len"] = out["title_len"]
        out["f_abs_len"] = out["abs_len"]
        out["f_year"] = out["year"]
        return out

    cand_train = prep(cand_train, qlen_map_train)
    cand_test  = prep(cand_test, qlen_map_test)

    # Labels joinen
    train_df = cand_train.merge(qrels, on=["qid", "docno"], how="left")
    train_df["label"] = train_df["label"].fillna(0).astype(int)
    train_df = train_df.sort_values(["qid", "rank"]).groupby("qid").head(TRAIN_TOPK).reset_index(drop=True)

    feature_cols = ["f_bm25", "f_qlen", "f_doclen", "f_title_len", "f_abs_len", "f_year"]
    X = train_df[feature_cols]
    y = train_df["label"]
    group_sizes = train_df.groupby("qid").size().tolist()

    print(f"TRAIN: rows={len(train_df)} | groups={len(group_sizes)} | features={feature_cols}")

    # LTR Model
    lgb_train = lgb.Dataset(X, label=y, group=group_sizes, free_raw_data=False)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": MODEL_LEARNING_RATE,
        "num_leaves": MODEL_NUM_LEAVES,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "seed": SEED,
    }
    model = lgb.train(params=params, train_set=lgb_train, num_boost_round=MODEL_NUM_TREES)
    print("MODEL: trained.")

    # BM25 Test Run schreiben
    bm25_test = cand_test.sort_values(["qid", "rank"]).reset_index(drop=True)
    bm25_test["new_rank"] = bm25_test.groupby("qid").cumcount()
    bm25_test["bm25_score"] = bm25_test["score"].astype(float)
    write_run_jsonl(bm25_test, RUN_BM25_TEST, "bm25_score")
    print(f"WROTE: {RUN_BM25_TEST.resolve()}")

    # LTR Rerank
    ltr_test = cand_test.copy()
    ltr_test["ltr_score"] = model.predict(ltr_test[feature_cols])
    ltr_test = ltr_test.sort_values(["qid", "ltr_score"], ascending=[True, False])
    ltr_test["new_rank"] = ltr_test.groupby("qid").cumcount()
    write_run_jsonl(ltr_test, RUN_LTR_TEST, "ltr_score")
    print(f"WROTE: {RUN_LTR_TEST.resolve()}")

    # === Ergebnis-Tabelle erstellen ===
    TOP_K = 10  # Anzahl Dokumente pro Query

    results_table = ltr_test.sort_values(["qid", "ltr_score"], ascending=[True, False])
    results_table["rank"] = results_table.groupby("qid").cumcount() + 1  # 1-basiert

    # Nur die Top K pro Query
    results_table = results_table[results_table["rank"] <= TOP_K]

    # Optional: nur relevante Spalten
    results_table = results_table[["qid", "docno", "rank", "ltr_score", "bm25_score", 
                                "f_doclen", "f_title_len", "f_abs_len", "f_year"]]

    # Zeige Tabelle an
    print("=== Top Ergebnisse pro Query ===")
    print(results_table.head(20))  # zeigt die ersten 20 Zeilen

    print("DONE.")





if __name__ == "__main__":
    main()
