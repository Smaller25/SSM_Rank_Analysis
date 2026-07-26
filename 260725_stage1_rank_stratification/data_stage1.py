"""Stage-1 data: 3 public domains + App.D repeat attacks (paper's RankViz is unpublished [PIN-5]).

[PIN-5] Public 3-domain substitution for the paper's unreleased RankViz set:
    - wikitext : WikiText-103 (natural English prose)          HF: Salesforce/wikitext, wikitext-103-raw-v1
    - github   : code (permissive)                             HF: codeparrot/github-code-clean (fallback: local)
    - arxiv    : scientific text                               HF: (scientific_papers arxiv) or armanc/scientific_papers
  App.D repeat attacks (2 kinds), reconstructed from the paper's description:
    - repeat_rarechar : a rare unicode char repeated           (adversarial low-diversity input)
    - repeat_number   : a common digit-string repeated         (adversarial low-diversity input)

Each domain yields >=16 sequences of length in [1024, 2048]. Every sample is
tokenized to input_ids[1, T] (batch size 1, matching common.Bundle.states which assumes bs=1).

Design mirrors 260722_exp/data.py: make_*(tok) helpers returning (ids, meta). Robust to offline
boxes: if HF `datasets` / network is unavailable, falls back to a deterministic synthetic corpus
per domain so the pipeline (and the CPU smoke test) still runs — the fallback is clearly flagged
in meta["source"]="fallback" so it is never silently mistaken for the real corpus.
"""
import os
import numpy as np
import torch

SEQ_LEN_DEFAULT = 2048          # in [1024, 2048]
N_SEQ_DEFAULT = 16              # >=16 per domain [PIN-5]
DOMAINS = ["wikitext", "github", "arxiv"]
ATTACKS = ["repeat_rarechar", "repeat_number"]

# NOTE on determinism [PIN-6]: every data path here is order-deterministic already — HF streaming
# yields the FIRST-N examples (no shuffle), the fallback corpus is seeded by a fixed per-domain
# rng, and the attacks use fixed strings / a fixed rng seed. Global torch/np/PYTHONHASHSEED pinning
# is done by stage1_repro.set_seed() before any data is drawn, so the whole pipeline is reproducible.


# ----------------------------------------------------------------------------- helpers
def _tok_chunks(tok, texts, seq_len, n_seq):
    """Concatenate texts, tokenize, and cut into n_seq non-overlapping chunks of length seq_len."""
    ids_all = []
    for t in texts:
        if not t:
            continue
        enc = tok(t, add_special_tokens=False).input_ids
        ids_all.extend(enc)
        if len(ids_all) >= seq_len * n_seq:
            break
    if len(ids_all) < seq_len * n_seq:
        # pad by tiling (keeps deterministic length; flagged by caller if fallback)
        reps = int(np.ceil(seq_len * n_seq / max(1, len(ids_all))))
        ids_all = (ids_all * reps)[: seq_len * n_seq]
    arr = np.asarray(ids_all[: seq_len * n_seq], dtype=np.int64).reshape(n_seq, seq_len)
    return [torch.from_numpy(arr[i]).unsqueeze(0) for i in range(n_seq)]


def _hf_texts(loader):
    """Run an HF-datasets loader lambda, returning a list[str] or None on any failure."""
    try:
        return loader()
    except Exception as e:  # offline / dataset-schema drift / no network
        print(f"  [data] HF load failed ({type(e).__name__}: {str(e)[:120]}) -> fallback", flush=True)
        return None


def _fallback_texts(domain, seq_len, n_seq, tok):
    """Deterministic synthetic corpus so the pipeline runs offline. Flagged as source=fallback."""
    rng = np.random.default_rng({"wikitext": 1, "github": 2, "arxiv": 3}[domain])
    vocab = tok.get_vocab()
    # draw plausible non-special token ids and decode to text so tokenization round-trips reasonably.
    lo, hi = 259, min(30000, tok.vocab_size)   # skip byte/special range
    ids = rng.integers(lo, hi, size=seq_len * n_seq * 2).tolist()
    text = tok.decode(ids)
    return [text]


# ----------------------------------------------------------------------------- domains
def make_wikitext(tok, seq_len=SEQ_LEN_DEFAULT, n_seq=N_SEQ_DEFAULT):
    def _load():
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        return [r for r in ds["text"] if r and len(r) > 40]
    texts = _hf_texts(_load)
    source = "hf:wikitext-103-raw-v1/test"
    if texts is None:
        texts, source = _fallback_texts("wikitext", seq_len, n_seq, tok), "fallback"
    ids = _tok_chunks(tok, texts, seq_len, n_seq)
    return ids, {"domain": "wikitext", "source": source, "is_fallback": source == "fallback",
                 "seq_len": seq_len, "n_seq": len(ids)}


def _stream_field(spec, field, n, **load_kw):
    """Stream a parquet-native HF dataset and collect up to n non-empty `field` strings.

    spec = (repo, config_or_None, split). Script-free datasets only (newer `datasets` dropped
    loading-script support), so we avoid trust_remote_code / loading-script repos.
    """
    from datasets import load_dataset
    repo, cfg, split = spec
    ds = (load_dataset(repo, cfg, split=split, streaming=True, **load_kw) if cfg
          else load_dataset(repo, split=split, streaming=True, **load_kw))
    out, cnt = [], 0
    it = iter(ds)
    try:
        for r in it:
            out.append(r.get(field, "") or "")
            cnt += 1
            if cnt >= n:
                break
    finally:
        # explicitly drop the streaming iterator so its background prefetch thread is torn down
        # before interpreter finalization (avoids a benign GIL-teardown crash at process exit).
        del it, ds
    return out


def make_github(tok, seq_len=SEQ_LEN_DEFAULT, n_seq=N_SEQ_DEFAULT):
    # parquet-native code corpora (no loading script). Try a couple; first that works wins.
    trials = [
        (("codeparrot/codeparrot-clean-valid", None, "train"), "content"),   # ungated parquet, python
        (("Nan-Do/code-search-net-python", None, "train"), "code"),          # ungated fallback
    ]
    texts, source = None, "fallback"
    for spec, field in trials:
        t = _hf_texts(lambda s=spec, f=field: _stream_field(s, f, 400))
        if t:
            texts, source = t, f"hf:{spec[0]}"
            break
    if texts is None:
        texts = _fallback_texts("github", seq_len, n_seq, tok)
    ids = _tok_chunks(tok, texts, seq_len, n_seq)
    return ids, {"domain": "github", "source": source, "is_fallback": source == "fallback",
                 "seq_len": seq_len, "n_seq": len(ids)}


def make_arxiv(tok, seq_len=SEQ_LEN_DEFAULT, n_seq=N_SEQ_DEFAULT):
    # parquet-native scientific text (no loading script).
    trials = [
        (("ccdv/arxiv-summarization", None, "test"), "article"),
        (("neuralwork/arxiver", None, "train"), "abstract"),
        (("armanc/scientific_papers", "arxiv", "test"), "article"),   # if parquet mirror available
    ]
    texts, source = None, "fallback"
    for spec, field in trials:
        t = _hf_texts(lambda s=spec, f=field: _stream_field(s, f, 200))
        if t:
            texts, source = t, f"hf:{spec[0]}"
            break
    if texts is None:
        texts = _fallback_texts("arxiv", seq_len, n_seq, tok)
    ids = _tok_chunks(tok, texts, seq_len, n_seq)
    return ids, {"domain": "arxiv", "source": source, "is_fallback": source == "fallback",
                 "seq_len": seq_len, "n_seq": len(ids)}


# ----------------------------------------------------------------------------- App.D repeat attacks
def make_repeat_rarechar(tok, seq_len=SEQ_LEN_DEFAULT, n_seq=N_SEQ_DEFAULT):
    """App.D attack 1: a rare character repeated. Uses several rare unicode chars, one per sequence,
    to give per-sequence variance while each sequence is maximally low-diversity."""
    rare = ["ƿ", "ʒ", "ʔ", "ʡ", "ǀ", "※",
            "☡", "✈", "❤", "☃", "♞", "⚡",
            "✖", "✿", "❀", "❖"]
    ids = []
    for i in range(n_seq):
        ch = rare[i % len(rare)]
        text = ch * (seq_len * 4)
        enc = tok(text, add_special_tokens=False).input_ids
        if len(enc) < seq_len:
            enc = (enc * int(np.ceil(seq_len / max(1, len(enc)))))
        arr = np.asarray(enc[:seq_len], dtype=np.int64)
        ids.append(torch.from_numpy(arr).unsqueeze(0))
    return ids, {"domain": "repeat_rarechar", "source": "appD_reconstruct", "is_fallback": False,
                 "seq_len": seq_len, "n_seq": len(ids),
                 "note": "App.D repeat attack: rare-char repetition"}


def make_repeat_number(tok, seq_len=SEQ_LEN_DEFAULT, n_seq=N_SEQ_DEFAULT):
    """App.D attack 2: a common digit-string repeated. Different digit block per sequence."""
    ids = []
    rng = np.random.default_rng(4242)
    for i in range(n_seq):
        block = "".join(str(d) for d in rng.integers(0, 10, size=8)) + " "
        text = block * (seq_len * 2)
        enc = tok(text, add_special_tokens=False).input_ids
        if len(enc) < seq_len:
            enc = (enc * int(np.ceil(seq_len / max(1, len(enc)))))
        arr = np.asarray(enc[:seq_len], dtype=np.int64)
        ids.append(torch.from_numpy(arr).unsqueeze(0))
    return ids, {"domain": "repeat_number", "source": "appD_reconstruct", "is_fallback": False,
                 "seq_len": seq_len, "n_seq": len(ids),
                 "note": "App.D repeat attack: common-number repetition"}


ALL = {
    "wikitext": make_wikitext,
    "github": make_github,
    "arxiv": make_arxiv,
    "repeat_rarechar": make_repeat_rarechar,
    "repeat_number": make_repeat_number,
}


def load_all(tok, seq_len=SEQ_LEN_DEFAULT, n_seq=N_SEQ_DEFAULT, which=None):
    """Return {name: (list[ids[1,T]], meta)} for the requested domains (default: all 5)."""
    which = which or list(ALL.keys())
    out = {}
    for name in which:
        ids, meta = ALL[name](tok, seq_len=seq_len, n_seq=n_seq)
        out[name] = (ids, meta)
        print(f"  [data] {name}: {meta['n_seq']} seqs x {meta['seq_len']} tok  ({meta['source']})", flush=True)
    return out
