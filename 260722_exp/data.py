"""GT 라벨이 있는 데이터셋 D1~D7. 각 make_*(tok) -> (input_ids[1,T], gt_boundaries|None, meta).

D1~D6: 원 노트북(260720_..§1B)과 동일. D7: RULER MK-NIAH(niah_ruler.make_mk_niah).
gt_boundaries = 정답 경계(토큰 위치). D1은 동질(경계 없음)이라 None(포화 캘리브레이션용).
"""
import torch


def _ids(tok, text, target=512):
    return tok(text, return_tensors="pt").input_ids[:, :target]


def _trim(tok, text, n):
    return tok.decode(tok(text, add_special_tokens=False).input_ids[:n])


def _concat_bounds(tok, parts):
    acc, b = "", []
    for i, p in enumerate(parts):
        acc += p
        if i < len(parts) - 1:
            b.append(tok(acc, return_tensors="pt").input_ids.shape[1])
    return tok(acc, return_tensors="pt").input_ids, b


def make_D1(tok):
    s = "The old lighthouse keeper watched the grey waves crash against the jagged rocks each night. "
    ii = _ids(tok, s * 60, 512)
    return ii, None, {"name": "D1_repeat", "note": "동질 반복 — 포화 지점 캘리브레이션"}


def make_D2(tok):
    pairs = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"),
             ("Egypt", "Cairo"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Canada", "Ottawa"),
             ("Norway", "Oslo"), ("Brazil", "Brasilia"), ("Greece", "Athens"), ("Cuba", "Havana"),
             ("Chile", "Santiago"), ("Ghana", "Accra"), ("Nepal", "Kathmandu"), ("Qatar", "Doha"),
             ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Oman", "Muscat"), ("Mali", "Bamako"),
             ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
             ("Hungary", "Budapest"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Turkey", "Ankara"),
             ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Colombia", "Bogota"), ("Angola", "Luanda"),
             ("Morocco", "Rabat"), ("Tunisia", "Tunis"), ("Jordan", "Amman"), ("Lebanon", "Beirut"),
             ("Denmark", "Copenhagen"), ("Belgium", "Brussels"), ("Switzerland", "Bern"), ("Iceland", "Reykjavik"),
             ("Croatia", "Zagreb"), ("Serbia", "Belgrade"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
             ("Ukraine", "Kyiv"), ("Russia", "Moscow"), ("China", "Beijing"), ("Pakistan", "Islamabad"),
             ("Bangladesh", "Dhaka"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"), ("Australia", "Canberra"),
             ("Cambodia", "Phnom Penh"), ("Mongolia", "Ulaanbaatar"), ("Bolivia", "Sucre"), ("Ecuador", "Quito"),
             ("Uganda", "Kampala"), ("Zambia", "Lusaka"), ("Senegal", "Dakar"), ("Latvia", "Riga"),
             ("Estonia", "Tallinn"), ("Slovakia", "Bratislava"), ("Georgia", "Tbilisi"), ("Armenia", "Yerevan")]
    sents = [f"The capital of {c} is {v}. " for c, v in pairs]
    ii, b = _concat_bounds(tok, sents)
    ii = ii[:, :512]; T = ii.shape[1]
    gt = [x for x in b if x < T]
    return ii, gt, {"name": "D2_structure", "note": "형식고정/내용변화 (dense GT)"}


def make_D3(tok):
    m = ("In number theory, a prime number is an integer greater than one whose only positive divisors "
         "are one and itself. The fundamental theorem of arithmetic states that every integer greater "
         "than one can be written uniquely as a product of primes, up to ordering. Euclid proved more "
         "than two thousand years ago that there are infinitely many primes, using a short argument by "
         "contradiction: assume the list of primes is finite, multiply them together, and add one, "
         "producing a number that no prime on the list can divide. The distribution of primes thins out "
         "among larger integers, and the prime number theorem describes how their density decreases "
         "logarithmically. The Riemann hypothesis, still unproven, concerns the precise fluctuations "
         "around that average. Modern cryptography relies on the difficulty of factoring the product of "
         "two very large primes, which keeps many public key systems secure across the internet today.")
    mu = ("In tonal harmony, the major scale follows a fixed pattern of whole and half steps that gives "
          "it a bright and stable character. Chords are built by stacking intervals of a third, so a "
          "basic triad contains a root, a third, and a fifth sounded together. A cadence is a sequence "
          "of chords that brings a phrase to a sense of rest, and the dominant seventh chord creates "
          "tension that pulls strongly back toward the tonic. Modulation shifts the music from one key "
          "to another, refreshing the harmonic color. Counterpoint weaves several independent melodic "
          "lines together, each keeping its own shape while fitting the whole. Composers combine rhythm, "
          "dynamics, phrasing, and timbre to shape a melody into a single unfolding idea over time.")
    co = ("In Python, a decorator is a callable that takes another function and returns a modified "
          "version of it, often used to add logging or timing without changing the original body. List "
          "comprehensions build lists concisely from an iterable in one readable expression, while "
          "generators yield items lazily to save memory on large sequences. Exceptions are handled with "
          "try and except blocks, letting a program recover from errors gracefully instead of crashing. "
          "Context managers, used with the with statement, acquire and release resources such as files "
          "automatically. Objects bundle data and behavior together, and classes define the blueprint "
          "from which instances are created. Type hints document expected argument and return types.")
    parts = [_trim(tok, m, 175) + " ", _trim(tok, mu, 175) + " ", _trim(tok, co, 175)]
    ii, gt = _concat_bounds(tok, parts)
    return ii, gt, {"name": "D3_topic", "note": "Math|Music|Code 전환"}


def make_D4(tok):
    en = ("Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide "
          "into glucose and oxygen. It takes place in the chloroplasts, where the pigment chlorophyll "
          "absorbs light energy. In the light-dependent reactions, this energy splits water molecules and "
          "releases oxygen. In the Calvin cycle, the captured energy is used to build sugar from carbon "
          "dioxide. The plant uses this sugar for growth and stores the extra as starch. Oxygen is "
          "released as a byproduct, sustaining most life on Earth. Because it removes carbon dioxide from "
          "the air, photosynthesis also plays a central role in regulating the planet's climate.")
    zh = ("光合作用是绿色植物利用阳光、水和二氧化碳合成葡萄糖和氧气的过程。它发生在叶绿体中，色素叶绿素在那里吸收光能。"
          "在光反应阶段，这种能量分解水分子并释放氧气。在卡尔文循环中，捕获的能量被用来由二氧化碳合成糖。"
          "植物利用这些糖来生长，并把多余的以淀粉的形式储存起来。氧气作为副产物被释放出来，维持着地球上大多数生命。"
          "由于它能从空气中去除二氧化碳，光合作用在调节地球气候方面也起着核心作用。")
    ko = ("광합성은 녹색 식물이 햇빛과 물과 이산화탄소를 이용해 포도당과 산소를 만들어 내는 과정이다. 이 과정은 엽록체에서 "
          "일어나며, 색소인 엽록소가 빛 에너지를 흡수한다. 명반응 단계에서 이 에너지는 물 분자를 분해하고 산소를 방출한다. "
          "캘빈 회로에서는 포획된 에너지가 이산화탄소로부터 당을 만드는 데 쓰인다. 식물은 이 당을 성장에 사용하고 남는 것은 "
          "녹말로 저장한다. 산소는 부산물로 방출되어 지구 생명 대부분을 지탱한다. 또한 공기 중 이산화탄소를 제거하기 때문에 "
          "광합성은 지구 기후 조절에도 핵심적인 역할을 한다.")
    parts = [_trim(tok, en, 160) + " ", _trim(tok, zh, 160) + " ", _trim(tok, ko, 160)]
    ii, gt = _concat_bounds(tok, parts)
    return ii, gt, {"name": "D4_language", "note": "내용고정/언어전환 En|Zh|Ko"}


def make_D5(tok, N=160, seed=0):
    import numpy as np
    g = np.random.default_rng(seed)
    keys = g.choice(4000, size=N, replace=False) + 1000
    vals = g.integers(0, 64, size=N) + 5000
    seq = np.empty(2 * N, dtype=np.int64); seq[0::2] = keys; seq[1::2] = vals
    order = g.permutation(N)
    ids = np.concatenate([seq, keys[order]])
    ii = torch.tensor(ids).unsqueeze(0)
    return ii, [2 * N], {"name": f"D5_mqar_N{N}", "note": "부하 sweep, GT=질의시작"}


def make_D6(tok):
    fill = "The committee reviewed the quarterly logistics report and noted no unusual activity. "
    names = ["Falcon", "Harbor", "Meadow", "Cobalt", "Juniper", "Zenith"]
    codes = ["4271", "9038", "1650", "7392", "5814", "2607"]
    acc, bnd, chunks = "", [], []
    for nm, cd in zip(names, codes):
        block = fill * 4 + f"Important: the access code for {nm} is {cd}. "
        acc += block; chunks.append(block)
        bnd.append(tok(acc, return_tensors="pt").input_ids.shape[1])
    query = fill * 3 + f"Question: the access code for {names[0]} is"
    ii = tok("".join(chunks) + query, return_tensors="pt").input_ids[:, :640]
    T = ii.shape[1]
    return ii, [b for b in bnd if b < T], {"name": "D6_niah_toy", "note": "수제 다중 needle"}


def make_D7(tok, max_seq_length=2048):
    """RULER MK-NIAH (niah_multikey_1: essay/words/numbers, 4 keys) — GT=needle 삽입 토큰위치."""
    from niah_ruler import make_mk_niah
    text, gt, answers, query, meta = make_mk_niah(tok, max_seq_length=max_seq_length)
    ii = tok(text, return_tensors="pt").input_ids
    meta2 = {"name": "D7_mk_niah_ruler", "note": "lm-eval-harness RULER multikey_1 동일", **meta}
    return ii, gt, meta2


ALL = {"D1": make_D1, "D2": make_D2, "D3": make_D3, "D4": make_D4,
       "D5": make_D5, "D6": make_D6, "D7": make_D7}
