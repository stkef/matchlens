import json

import numpy as np
import pandas as pd
import pytest

from matchlens import metrics
from matchlens.data import group_split, load_split_frame, save_split
from matchlens.evaluate import run
from matchlens.retrievers import BM25Retriever, Candidates, drop_self
from matchlens.synthetic import make_synthetic
from matchlens.text import decode_title, tokenize
from matchlens.threshold import tune_threshold


# --- text ---------------------------------------------------------------------------------------

def test_tokenize_glues_units_to_numbers():
    assert tokenize("Samsung A52 128 GB") == tokenize("samsung a52 128gb") == ["samsung", "a52", "128gb"]
    assert tokenize("Serum 30 ML") == ["serum", "30ml"]
    assert tokenize("Rice Cooker 1,8 L") == ["rice", "cooker", "1.8l"]


def test_decode_title_undoes_byte_escapes():
    assert decode_title(r"Paket \xe2\x80\x9cHemat\xe2\x80\x9d") == "Paket “Hemat”"
    assert decode_title("plain &amp; simple") == "plain & simple"


# --- split --------------------------------------------------------------------------------------

def test_group_split_keeps_groups_whole_and_is_deterministic():
    df = make_synthetic(n_models=80, seed=1)
    a, b = group_split(df, seed=7), group_split(df, seed=7)
    assert a.equals(b)
    assert (df.assign(split=a).groupby("label_group")["split"].nunique() == 1).all()
    share = df.assign(split=a).drop_duplicates("label_group")["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.7, abs=0.02)


def test_save_split_refuses_to_silently_change(tmp_path):
    df = make_synthetic(n_models=20)
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "split.csv"
    save_split(df, group_split(df, seed=1), out, source_csv=csv, seed=1, fractions=(0.7, 0.1, 0.2))
    save_split(df, group_split(df, seed=1), out, source_csv=csv, seed=1, fractions=(0.7, 0.1, 0.2))  # same: ok
    with pytest.raises(FileExistsError):
        save_split(df, group_split(df, seed=2), out, source_csv=csv, seed=2, fractions=(0.7, 0.1, 0.2))
    assert json.loads(out.with_suffix(".json").read_text())["seed"] == 1


# --- metrics ------------------------------------------------------------------------------------

@pytest.fixture
def toy():
    # Rows 0,1,2 are group A; rows 3,4 are group B.
    labels = np.array([1, 1, 1, 2, 2])
    cands = Candidates(
        indices=np.array([[3, 1, 2], [0, 2, -1], [4, 3, 0], [4, 0, -1], [1, 3, -1]]),
        scores=np.array([[0.9, 0.8, 0.4], [0.9, 0.7, -np.inf], [0.6, 0.5, 0.3], [0.95, 0.2, -np.inf],
                         [0.4, 0.3, -np.inf]]),
    )
    return cands, labels


def test_recall_and_mrr(toy):
    cands, labels = toy
    # recall@2: row0 1/2, row1 2/2, row2 0/2, row3 1/1, row4 1/1
    assert metrics.recall_at_k(cands, labels, 2) == pytest.approx((0.5 + 1 + 0 + 1 + 1) / 5)
    # first true match ranks: 2, 1, 3, 1, 2
    assert metrics.mrr(cands, labels) == pytest.approx((1 / 2 + 1 + 1 / 3 + 1 + 1 / 2) / 5)


def test_f1_counts_self_like_the_competition(toy):
    cands, labels = toy
    f1 = metrics.f1_per_listing(cands, labels, threshold=0.75)
    # row0 predicts {0,3,1}: tp=2, |P|=3, |T|=3 -> 4/6 ; row4 predicts only itself: tp=1, |P|=1, |T|=2 -> 2/3
    assert f1[0] == pytest.approx(4 / 6)
    assert f1[4] == pytest.approx(2 / 3)
    # Threshold above every score: each listing predicts only itself.
    assert metrics.f1_per_listing(cands, labels, 2.0) == pytest.approx(2 / (1 + np.array([3, 3, 3, 2, 2])))


def test_precision_at_recall(toy):
    cands, labels = toy
    # 8 true ordered pairs exist; retrieved true pairs by score: .9 .8 .95 .9 .7 .3 .3 -> 7/8 max recall.
    assert np.isnan(metrics.precision_at_recall(cands, labels, 0.9))
    # Sorted: .95T .9F .9T .8T .7T ... -> recall 0.5 (4 of 8) first reached after 5 pairs, 4 true.
    assert metrics.precision_at_recall(cands, labels, 0.5) == pytest.approx(4 / 5)
    perfect = Candidates(np.array([[1], [0]]), np.array([[0.9], [0.9]]))
    assert metrics.precision_at_recall(perfect, np.array([5, 5]), 0.9) == 1.0
    missing = Candidates(np.array([[-1], [-1]]), np.array([[-np.inf], [-np.inf]]))
    assert np.isnan(metrics.precision_at_recall(missing, np.array([5, 5]), 0.9))


def test_metrics_with_distractor_rows_after_the_queries():
    # Two queries (rows 0,1, group 1); rows 2,3 are distractors from group 9 and are never queried.
    labels = np.array([1, 1, 9, 9])
    cands = Candidates(np.array([[2, 1], [3, 0]]), np.array([[0.9, 0.8], [0.7, 0.6]]))
    assert metrics.recall_at_k(cands, labels, 1) == 0.0
    assert metrics.recall_at_k(cands, labels, 2) == 1.0
    assert metrics.mrr(cands, labels) == pytest.approx(0.5)
    # Threshold 0.85 lets in only the distractor for query 0: tp=1, |P|=2, |T|=2 -> 0.5; query 1 -> 2/3.
    assert metrics.f1_per_listing(cands, labels, 0.85).tolist() == pytest.approx([0.5, 2 / 3])


def test_threshold_tuner_finds_the_separating_cut():
    labels = np.array([1, 1, 2, 2])
    # True matches score 0.9, the look-alikes score 0.5: any threshold in (0.5, 0.9] is perfect.
    cands = Candidates(np.array([[1, 2], [0, 3], [3, 0], [2, 1]]),
                       np.array([[0.9, 0.5], [0.9, 0.5], [0.9, 0.5], [0.9, 0.5]]))
    result = tune_threshold(cands, labels)
    assert result.f1 == pytest.approx(1.0)
    assert 0.5 < result.threshold <= 0.9


# --- BM25 ---------------------------------------------------------------------------------------

def test_bm25_ranks_exact_spec_first_and_normalises_self_to_one():
    corpus = pd.DataFrame({"title": [
        "Samsung Galaxy A52 128GB hitam",
        "Samsung Galaxy A52 64GB hitam",
        "Xiaomi Redmi Note 10 128GB",
        "Wardah lip cream 30ml",
    ]})
    r = BM25Retriever()
    r.fit(corpus)
    res = r.search(pd.DataFrame({"title": ["galaxy a52 128 gb"]}), k=4)
    assert res.indices[0, 0] == 0
    assert res.indices[0, 1] == 1
    self_res = r.search(corpus.iloc[[0]], k=1)
    assert self_res.indices[0, 0] == 0 and self_res.scores[0, 0] == pytest.approx(1.0)
    assert -1 in res.indices[0]  # the lip cream shares no token and is not a candidate


def test_drop_self_removes_only_the_query_row():
    c = Candidates(np.array([[0, 2, 1], [0, 1, 2]]), np.array([[1.0, 0.5, 0.4], [0.7, 1.0, 0.2]]))
    out = drop_self(c, 2)
    assert out.indices.tolist() == [[2, 1], [0, 2]]


# --- end to end ---------------------------------------------------------------------------------

def test_evaluate_end_to_end_and_test_lock(tmp_path):
    df = make_synthetic(n_models=60, seed=3)
    csv, split_path = tmp_path / "train.csv", tmp_path / "split.csv"
    df.to_csv(csv, index=False)
    save_split(df, group_split(df, (0.6, 0.2, 0.2), seed=0), split_path, source_csv=csv, seed=0,
               fractions=(0.6, 0.2, 0.2))
    cfg = {"name": "t", "rung": 1, "data": {"csv": str(csv), "split": str(split_path)},
           "retriever": {"type": "bm25"}, "eval": {"k": 50, "latency_sample": 10}}
    results = tmp_path / "results"

    with pytest.raises(PermissionError):
        run(cfg, "test", results)
    val = run(cfg, "val", results)["metrics"]
    assert val["n_queries"] == len(load_split_frame(csv, split_path, "val"))
    assert 0 < val["f1"] <= 1 and 0 < val["recall@50"] <= 1
    test = run(cfg, "test", results, unlock_test=True)["metrics"]
    assert test["threshold"] == val["threshold"]
    ledger = pd.read_csv(results / "ledger.csv")
    assert ledger["split"].tolist() == ["val", "test"]

    # Distractors grow the pool but not the query set, and can only make retrieval harder.
    hard = run({**cfg, "name": "t_hard", "eval": {**cfg["eval"], "distractors": ["train"]}}, "val", results)
    assert hard["metrics"]["n_queries"] == val["n_queries"]
    assert hard["metrics"]["pool_size"] == val["n_queries"] + len(load_split_frame(csv, split_path, "train"))
    assert hard["metrics"]["recall@50"] <= val["recall@50"]
    with pytest.raises(ValueError):
        run({**cfg, "eval": {**cfg["eval"], "distractors": ["test"]}}, "val", results)


def test_canonical_units_unify_spellings_and_scale():
    assert tokenize("Profeline 400 gram", canonical_units=True) == tokenize("profeline 400gr", True) == ["profeline", "400g"]
    assert tokenize("Beras 1 kg", True) == tokenize("Beras 1000 gr", True) == ["beras", "1000g"]
    assert tokenize("Minyak 1,8 L", True) == ["minyak", "1800ml"]
    assert tokenize("Samsung A52 128 GB", True) == ["samsung", "a52", "128gb"]  # other units untouched
    assert tokenize("Profeline 400 gram") == ["profeline", "400gram"]  # default (rung 1) unchanged


def test_phash_boost_adds_near_identical_photos_as_certain_matches():
    from matchlens.retrievers import build_retriever
    corpus = pd.DataFrame({
        "title": ["kaos polos hitam", "baju anak", "sepatu lari"],
        "image_phash": ["ffff0000ffff0000", "ffff0000ffff0001", "0000ffff0000ffff"],  # 0 and 1 differ by 1 bit
    })
    r = build_retriever({"type": "phash_boost", "max_dist": 2, "base": {"type": "bm25"}})
    r.fit(corpus)
    res = drop_self(r.search(corpus, 3), 2)
    assert res.indices[0, 0] == 1 and res.scores[0, 0] == 1.0  # no shared words, but same photo
    assert 2 not in res.indices[0]                             # 32 bits apart: not a near-duplicate


def test_dense_retriever_uses_cosine_over_precomputed_vectors(tmp_path):
    from matchlens.retrievers import build_retriever
    vecs = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float16)
    np.savez(tmp_path / "emb.npz", posting_id=np.array(["a", "b", "c"]), vectors=vecs)
    corpus = pd.DataFrame({"posting_id": ["a", "b", "c"], "title": ["x", "y", "z"]})
    r = build_retriever({"type": "dense", "path": str(tmp_path / "emb.npz")})
    r.fit(corpus)
    res = r.search(corpus.iloc[[0]], 3)
    assert res.indices[0].tolist() == [0, 1, 2]
    assert res.scores[0].tolist() == pytest.approx([1.0, 0.8, 0.0], abs=1e-3)
    with pytest.raises(KeyError):
        r.search(pd.DataFrame({"posting_id": ["zzz"], "title": ["?"]}), 1)


def test_vector_stores_agree_with_exact_search(tmp_path):
    from matchlens.stores import build_store
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(300, 16)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = corpus[:20]
    exact = build_store({"type": "numpy"})
    exact.build(corpus)
    ref_idx, ref_sc = exact.search(queries, 5)
    for cfg in [{"type": "faiss", "index": "flat", "path": str(tmp_path / "t.faiss")},
                {"type": "qdrant", "path": str(tmp_path / "qdrant"), "collection": "image_test"}]:
        store = build_store(cfg)
        store.build(corpus)
        idx, sc = store.search(queries, 5)
        assert (idx == ref_idx).all(), cfg["type"]
        assert sc == pytest.approx(ref_sc, abs=1e-4)
        if hasattr(store, "close"):
            store.close()
    # A saved FAISS index is reused when the vectors are unchanged.
    reused = build_store({"type": "faiss", "index": "flat", "path": str(tmp_path / "t.faiss")})
    reused.build(corpus)
    assert (reused.search(queries, 5)[0] == ref_idx).all()
