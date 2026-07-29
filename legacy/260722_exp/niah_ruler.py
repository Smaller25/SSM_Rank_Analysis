"""MK-NIAH (D7) — lm-evaluation-harness RULER niah_multikey_1 과 '정확히 동일한' 생성 로직.

출처: lm_eval/tasks/ruler/prepare_niah.py, niah_utils.py (NVIDIA RULER, Apache-2.0). 상수·함수·문구를
그대로 복제하고, needle 삽입 '토큰 위치'를 GT 경계로 뽑는 make_mk_niah() 래퍼만 추가함
(텍스트 생성 절차/파라미터는 harness와 동일; 재현을 위해 seed만 고정).

의존: wonderwords, nltk(punkt_tab), datasets(+ 'baber/paul_graham_essays'), 그리고 length 제어용 tokenizer.
  pip install wonderwords nltk datasets ; python -c "import nltk; nltk.download('punkt_tab')"
"""
import random
import re
import uuid
from functools import lru_cache, cache
from typing import List, Union, Literal

# ── RULER 상수/문구 (verbatim) ────────────────────────────────────────────
NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
TEMPLATE = ("Some special magic {type_needle_v} are hidden within the following text. Make sure to "
            "memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are "
            "all the special magic {type_needle_v} for {query} mentioned in the provided text?")
RANDOM_SEED = 42
try:
    import numpy as _np
    DEPTHS = list(_np.round(_np.linspace(0, 100, num=40, endpoint=True)).astype(int))
except Exception:
    DEPTHS = list(range(0, 101, 100 // 39))


@cache
def _get_words() -> List[str]:
    import wonderwords
    r = wonderwords.RandomWord()
    nouns = r._categories["nouns"]; adjs = r._categories["adjectives"]
    words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
    return sorted(list(set(words)))


@lru_cache(maxsize=1024)
def _sent_tokenize(text: str) -> List[str]:
    from nltk import sent_tokenize
    return sent_tokenize(text)


def _rand_number(num_digits=7) -> str:
    return str(random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1))


def _rand_word() -> str:
    return random.choice(_get_words())


def _rand_uuid() -> str:
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def _rand(type_needle: str) -> str:
    return {"numbers": _rand_number, "words": _rand_word, "uuids": _rand_uuid}[type_needle]()


@cache
def get_haystack(type_haystack: Literal["essay", "repeat", "needle"]) -> Union[List[str], str]:
    if type_haystack == "essay":
        import datasets
        essay = datasets.load_dataset("baber/paul_graham_essays", split="train")["text"]
        essay = " ".join(essay)
        return re.sub(r"\s+", " ", essay).split(" ")
    if type_haystack == "repeat":
        return "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    if type_haystack == "needle":
        return NEEDLE
    raise NotImplementedError(type_haystack)


def generate_input_output_gt(num_haystack, haystack, *, type_haystack, num_needle_k, type_needle_k,
                             num_needle_v, type_needle_v, template, num_needle_q=1,
                             random_seed=RANDOM_SEED):
    """generate_input_output 의 복제 + needles 리스트도 함께 반환 (텍스트 구성은 동일)."""
    keys, values, needles = [], [], []
    for _ in range(num_needle_k):
        keys.append(_rand(type_needle_k))
        value = []
        for _ in range(num_needle_v):
            value.append(_rand(type_needle_v))
            needles.append(NEEDLE.format(type_needle_v=type_needle_v, key=keys[-1], value=value[-1]))
        values.append(value)
    random.Random(random_seed).shuffle(needles)

    if type_haystack == "essay":
        assert isinstance(haystack, list)
        text = " ".join(haystack[:num_haystack])
        document_sents = _sent_tokenize(text.strip())
        insertion_positions = ([0]
                               + sorted([int(len(document_sents) * (d / 100))
                                         for d in random.sample(DEPTHS, len(needles))])
                               + [len(document_sents)])
        document_sents_list = []
        for i in range(1, len(insertion_positions)):
            last_pos, next_pos = insertion_positions[i - 1], insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i - 1 < len(needles):
                document_sents_list.append(needles[i - 1])
        context = " ".join(document_sents_list)
    else:
        if type_haystack == "repeat":
            sentences = [haystack] * num_haystack
        elif type_haystack == "needle":
            sentences = [haystack.format(type_needle_v=type_needle_v, key=_rand(type_needle_k),
                                         value=_rand(type_needle_v)) for _ in range(num_haystack)]
        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)

    indices = random.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = (", ".join(queries[:-1]) + ", and " + queries[-1]) if len(queries) > 1 else queries[0]

    if num_needle_q * num_needle_v == 1:
        template = template.replace("Some", "A").replace("are all", "is").replace("are", "is")
        template = template.replace("answers", "answer")
        type_needle_v = type_needle_v[:-1]

    input_text = template.format(type_needle_v=type_needle_v, context=context, query=query)
    return input_text, answers, query, needles


def make_mk_niah(tok, max_seq_length=2048, *, type_haystack="essay", type_needle_k="words",
                 type_needle_v="numbers", num_needle_k=4, num_needle_v=1, num_needle_q=1,
                 tokens_to_generate=128, seed=RANDOM_SEED):
    """RULER niah_multikey_1 파라미터(기본)로 한 샘플 생성 + needle 시작 '토큰 위치'를 GT로 반환.

    반환: (input_text, gt_token_positions[list], answers, query, meta)
    """
    random.seed(seed)
    haystack = get_haystack(type_haystack)
    num_needle_k = max(num_needle_k, num_needle_q)
    incremental = 500 if type_haystack == "essay" else 25
    if type_haystack != "essay" and max_seq_length < 4096:
        incremental = 5
    num_haystack = incremental

    # generate_samples 의 num_haystack 탐색 로직 (동일)
    total_tokens = 0
    while total_tokens + tokens_to_generate < max_seq_length:
        txt, ans, q, _ = generate_input_output_gt(
            num_haystack, haystack, type_haystack=type_haystack, num_needle_k=num_needle_k,
            type_needle_k=type_needle_k, num_needle_v=num_needle_v, type_needle_v=type_needle_v,
            template=TEMPLATE, num_needle_q=num_needle_q, random_seed=seed)
        total_tokens = len(tok(txt + " ".join(ans)).input_ids)
        if total_tokens + tokens_to_generate > max_seq_length:
            num_haystack = max(incremental, num_haystack - incremental)
            break
        if type_haystack == "essay" and num_haystack > len(haystack):
            num_haystack = len(haystack)
            break
        num_haystack += incremental

    # 최종 샘플 (길이 초과 시 haystack 줄이며 재시도 — 동일)
    used = num_haystack
    while True:
        try:
            input_text, answers, query, needles = generate_input_output_gt(
                used, haystack, type_haystack=type_haystack, num_needle_k=num_needle_k,
                type_needle_k=type_needle_k, num_needle_v=num_needle_v, type_needle_v=type_needle_v,
                template=TEMPLATE, num_needle_q=num_needle_q, random_seed=seed)
            if len(tok(input_text).input_ids) + tokens_to_generate <= max_seq_length:
                break
        except Exception:
            pass
        if used > incremental:
            used -= incremental
        else:
            break

    # needle 시작 토큰 위치 = GT 경계
    gt = []
    for nd in needles:
        idx = input_text.find(nd)
        if idx >= 0:
            gt.append(len(tok(input_text[:idx]).input_ids))
    gt = sorted(set(gt))
    meta = {"num_haystack": used, "n_needles": len(needles), "answers": answers, "query": query,
            "n_tokens": len(tok(input_text).input_ids)}
    return input_text, gt, answers, query, meta
