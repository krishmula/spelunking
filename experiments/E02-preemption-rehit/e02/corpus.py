
import hashlib
import json
import os
import random
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

MODEL = "mlx-community/Qwen3-4B-4bit"
VERSION = "v1"
SIZE = 100
LENGTH_RANGE = (960, 1088)
TARGET_TOKENS = 1024
BLOCK_SIZE = 16
SEED = 20261020
MIN_ROWS = 3
MAX_ROWS = 60

TEMPLATES = [
    "How many positive integers less than {n} are divisible by {a} or {b} but not by {c}?",
    "Find the remainder when {a}^{e} is divided by {m}.",
    "A bag has {r} red, {g} green and {b} blue marbles. You draw {k} marbles without replacement. "
    "What is the probability that exactly {j} of them are red? Give the answer as a reduced fraction.",
    "Find all integer pairs (x, y) with x^2 + {k}*y^2 = {n}.",
    "In how many ways can {n} people sit around a round table if {a} particular people must not sit next to "
    "each other? Rotations count as the same seating.",
    "What is the sum of the digits of all the integers from 1 to {n}?",
    "A sequence starts a_1 = {a} and continues a_(n+1) = (a_n^2 + {b}*a_n + {c}) mod {m}. Find a_{k}.",
    "Two trains start {d} km apart and drive toward each other at {v1} km/h and {v2} km/h. A bird flies back "
    "and forth between them at {v3} km/h until they meet. How far does the bird fly, and how many full trips "
    "does it complete if each trip takes it from one train to the other?",
]

NAMES = [
    "Amara", "Bjorn", "Chidi", "Dalia", "Emeka", "Farah", "Gustav", "Hana", "Ivo", "Jun", "Kavya", "Lior",
    "Mateo", "Nadia", "Oskar", "Priya", "Quinn", "Rosa", "Sanjay", "Tomas", "Ulla", "Vikram", "Wen", "Ximena",
    "Yusuf", "Zofia", "Anika", "Bruno", "Celine", "Dmitri", "Esme", "Femi", "Greta", "Hiro", "Ines", "Jonas",
]


class CorpusError(Exception):
    pass


@dataclass(frozen=True)
class CorpusItem:
    item: int
    template: int
    text: str
    token_ids: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class Corpus:
    version: str
    model: str
    tokenizer_revision: str
    chat_template_sha256: str
    transformers_version: str
    generator_sha256: str
    items: tuple[CorpusItem, ...]
    digest: str
    census: dict


def token_sha256(token_ids) -> str:
    return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()



def _row0(rng):
    return dict(n=rng.randrange(500, 5000), a=rng.randrange(3, 12), b=rng.randrange(13, 30), c=rng.randrange(31, 60))


def _row1(rng):
    return dict(a=rng.randrange(3, 12), e=rng.randrange(100, 2000), m=rng.randrange(13, 200))


def _row2(rng):
    return dict(r=rng.randrange(3, 9), g=rng.randrange(3, 9), b=rng.randrange(3, 9), k=rng.randrange(3, 7),
                j=rng.randrange(1, 3))


def _row3(rng):
    return dict(k=rng.randrange(3, 7), n=rng.randrange(500, 5000))


def _row4(rng):
    n = rng.randrange(7, 31)
    return dict(n=n, a=rng.randrange(2, min(5, n // 2 + 1)))


def _row5(rng):
    return dict(n=rng.randrange(500, 5000))


def _row6(rng):
    return dict(a=rng.randrange(3, 12), b=rng.randrange(13, 30), c=rng.randrange(31, 60), m=rng.randrange(13, 200),
                k=rng.randrange(5, 41))


def _row7(rng):
    return dict(d=rng.randrange(100, 400), v1=rng.randrange(40, 90), v2=rng.randrange(40, 90),
                v3=rng.randrange(100, 200))


SPECS = [
    dict(
        sample=_row0,
        row="Range {i}: positive integers less than {n}; count those divisible by {a} or by {b} but not by {c}.",
        opening="{r1[n]}, {r1[a]}, {r1[b]} and {r1[c]} open {name}'s divisibility audit of {rows} integer ranges.",
        premises=(
            "Each range starts at 1 and stops just below its bound. A number qualifies when at least one of the "
            "two listed divisors divides it and the excluded divisor does not. The divisors in a range need not "
            "be coprime, so use least common multiples in any inclusion-exclusion step."
        ),
        question=(
            "Give the qualifying count for every range, then the total over all ranges and the range with the "
            "largest count."
        ),
    ),
    dict(
        sample=_row1,
        row="Item {i}: the remainder when {a}^{e} is divided by {m}.",
        opening="{r1[a]}^{r1[e]} mod {r1[m]} is the first of {rows} modular powers that {name} must reduce.",
        premises=(
            "Every remainder lies between 0 and the modulus minus 1. Moduli may be composite and may share a "
            "factor with the base, so Euler's theorem applies only when base and modulus are coprime; otherwise "
            "split the modulus with the Chinese remainder theorem or track the cycle of powers directly."
        ),
        question=(
            "Find the remainder for every item, then report the sum of all remainders and list any items whose "
            "remainders are equal."
        ),
    ),
    dict(
        sample=_row2,
        row=(
            "Bag {i}: {r} red, {g} green and {b} blue marbles; draw {k} without replacement; probability that "
            "exactly {j} are red."
        ),
        opening=(
            "{r1[r]}/{r1[g]}/{r1[b]} and {r2[r]}/{r2[g]}/{r2[b]} red/green/blue marbles fill the first two of "
            "{name}'s {rows} bags."
        ),
        premises=(
            "In every bag all marbles are equally likely to be drawn and the order of drawing does not matter, "
            "so each probability is a ratio of binomial coefficients. Marbles of one colour are interchangeable."
        ),
        question=(
            "Give each bag's probability as a reduced fraction, then name the bag with the highest probability "
            "and the bag with the lowest."
        ),
    ),
    dict(
        sample=_row3,
        row="Equation {i}: all integer pairs (x, y) with x^2 + {k}*y^2 = {n}.",
        opening="{r1[n]} = x^2 + {r1[k]}*y^2 is the first of {rows} equations in {name}'s lattice-point search.",
        premises=(
            "Solutions include negative values and zero, so (x, y), (-x, y), (x, -y) and (-x, -y) are distinct "
            "pairs when the entries are nonzero. Bound y by the square root of n divided by the coefficient, and "
            "use residues modulo small numbers to rule out equations with no solutions."
        ),
        question=(
            "List every solution pair for each equation, then report the total number of pairs across all "
            "equations and which equations have none."
        ),
    ),
    dict(
        sample=_row4,
        row=(
            "Table {i}: {n} people sit around a round table and {a} particular people must not sit next to each "
            "other; count the seatings."
        ),
        opening=(
            "{r1[n]}, {r2[n]} and {r3[n]} guests, with {r1[a]}, {r2[a]} and {r3[a]} feuding, fill the first three "
            "of {name}'s {rows} round tables."
        ),
        premises=(
            "Rotations of a seating count as the same seating, but reflections count as different. The listed "
            "people must be pairwise non-adjacent: no two of them may occupy neighbouring seats. All other "
            "guests are distinct and unrestricted."
        ),
        question="Give the number of seatings for every table, then the sum of all the counts.",
    ),
    dict(
        sample=_row5,
        row="Run {i}: the sum of the digits of all the integers from 1 to {n}.",
        opening="{r1[n]} is where the first of {name}'s {rows} digit-sum runs stops.",
        premises=(
            "Each run writes every integer from 1 up to and including its endpoint in base ten without leading "
            "zeros, then adds every digit written. Blocks of consecutive integers that cover each last digit "
            "0 to 9 equally contribute 45 per full cycle in that position."
        ),
        question="Give the digit sum for every run, then the total over all runs.",
    ),
    dict(
        sample=_row6,
        row="Sequence {i}: a_1 = {a}, a_(n+1) = (a_n^2 + {b}*a_n + {c}) mod {m}; find a_{k}.",
        opening=(
            "{r1[m]}, {r1[c]}, {r1[b]} and {r1[a]} are the modulus, constant, linear coefficient and first term "
            "of the first of {name}'s {rows} sequences."
        ),
        premises=(
            "Every term after the first is a residue between 0 and its modulus minus 1. Because each sequence "
            "takes finitely many values, it must eventually repeat; once a value recurs, the terms cycle with a "
            "fixed period, which can shorten the work for later indices."
        ),
        question="Find the requested term of every sequence, then report the sum of those terms.",
    ),
    dict(
        sample=_row7,
        row=(
            "Pair {i}: trains {d} km apart approach at {v1} km/h and {v2} km/h; the bird flies at {v3} km/h; "
            "find its distance and its number of full trips."
        ),
        opening="{r1[d]} km at {r1[v1]} and {r1[v2]} km/h describes the first of {name}'s {rows} train pairs.",
        premises=(
            "In each pair the trains move at constant speed on one straight track and the bird starts at the "
            "front of the first train at the moment both trains depart. It turns instantly on reaching a train. "
            "One trip is one flight from a train to the other train."
        ),
        question=(
            "For every pair give the bird's total distance and its number of full trips, then the total distance "
            "over all pairs."
        ),
    ),
]


def _text(spec, name, rows):
    opening = spec["opening"].format(name=name, rows=len(rows), r1=rows[0], r2=rows[1], r3=rows[2])
    lines = [opening, "", spec["premises"], ""]
    lines += [spec["row"].format(i=i + 1, **row) for i, row in enumerate(rows)]
    lines += ["", spec["question"]]
    return "\n".join(lines)


def _tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def _render(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    return tuple(tok(s, add_special_tokens=False)["input_ids"])


def _make_item(tok, i, openings):
    template = i % len(TEMPLATES)
    spec = SPECS[template]
    rng = random.Random(f"{SEED}:{i}")
    while True:
        name = rng.choice(NAMES)
        pool = []
        while len(pool) < MAX_ROWS:
            row = spec["sample"](rng)
            if row not in pool:
                pool.append(row)
        if _text(spec, name, pool[:MIN_ROWS])[:16] not in openings:
            break
    best = None
    for count in range(MIN_ROWS, MAX_ROWS + 1):
        text = _text(spec, name, pool[:count])
        ids = _render(tok, text)
        if LENGTH_RANGE[0] <= len(ids) <= LENGTH_RANGE[1]:
            if best is None or abs(len(ids) - TARGET_TOKENS) < abs(len(best[1]) - TARGET_TOKENS):
                best = (text, ids)
        if len(ids) > LENGTH_RANGE[1]:
            break
    if best is None:
        raise CorpusError(f"item {i}: no row count lands in {LENGTH_RANGE}")
    text, ids = best
    openings.add(text[:16])
    return CorpusItem(item=i, template=template, text=text, token_ids=ids, sha256=token_sha256(ids))


def _revision():
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL, local_files_only=True)).name


def _jsonl(items):
    return "".join(
        json.dumps({"item": it.item, "template": it.template, "text": it.text, "token_ids": list(it.token_ids),
                    "n_tokens": len(it.token_ids), "sha256": it.sha256}, sort_keys=True, separators=(",", ":"))
        + "\n"
        for it in items
    ).encode()


def _length_stats(items):
    lengths = [len(it.token_ids) for it in items]
    return {"length_min": min(lengths), "length_max": max(lengths), "length_mean": statistics.mean(lengths),
            "length_median": statistics.median(lengths)}


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def census(items) -> dict:
    items = list(items)
    prefixes = [_common_prefix(items[p].token_ids, items[q].token_ids)
                for p in range(len(items)) for q in range(p + 1, len(items))]
    hist = {}
    for n in prefixes:
        hist[str(n)] = hist.get(str(n), 0) + 1
    return {
        "pairs": len(prefixes),
        "max_common_prefix_tokens": max(prefixes, default=0),
        "common_prefix_hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "max_full_block_shared": max(prefixes, default=0) // BLOCK_SIZE,
        "pairs_with_full_block_shared": sum(1 for n in prefixes if n >= BLOCK_SIZE),
        **_length_stats(items),
    }


def _check_items(items):
    if len(items) != SIZE:
        raise CorpusError(f"expected {SIZE} items, found {len(items)}")
    for i, it in enumerate(items):
        if it.item != i or it.template != i % len(TEMPLATES):
            raise CorpusError(f"item {i}: wrong item or template index ({it.item}, {it.template})")
        if not LENGTH_RANGE[0] <= len(it.token_ids) <= LENGTH_RANGE[1]:
            raise CorpusError(f"item {i}: {len(it.token_ids)} tokens outside {LENGTH_RANGE}")
        if it.sha256 != token_sha256(it.token_ids):
            raise CorpusError(f"item {i}: sha256 does not match its token ids")
    if len({it.text for it in items}) != SIZE or len({it.token_ids for it in items}) != SIZE:
        raise CorpusError("items are not distinct")
    if len({it.text[:16] for it in items}) != SIZE:
        raise CorpusError("two items share their first 16 characters")


def build() -> Corpus:
    import transformers

    tok = _tokenizer()
    openings = set()
    items = tuple(_make_item(tok, i, openings) for i in range(SIZE))
    _check_items(items)
    return Corpus(
        version=VERSION,
        model=MODEL,
        tokenizer_revision=_revision(),
        chat_template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        transformers_version=transformers.__version__,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        items=items,
        digest=hashlib.sha256(_jsonl(items)).hexdigest(),
        census=census(items),
    )


def _paths(directory):
    directory = Path(directory)
    return directory / f"corpus-{VERSION}.jsonl", directory / f"corpus-{VERSION}.manifest.json"


def _manifest(corpus):
    stats = _length_stats(corpus.items)
    return (json.dumps({
        "version": corpus.version, "model": corpus.model, "tokenizer_revision": corpus.tokenizer_revision,
        "chat_template_sha256": corpus.chat_template_sha256, "transformers_version": corpus.transformers_version,
        "generator_sha256": corpus.generator_sha256, "size": len(corpus.items), "length_range": list(LENGTH_RANGE),
        "target_tokens": TARGET_TOKENS, "digest": corpus.digest, "census": corpus.census,
        "length_stats": {"min": stats["length_min"], "max": stats["length_max"], "mean": stats["length_mean"],
                         "median": stats["length_median"]},
        "seed": SEED,
    }, sort_keys=True, indent=1) + "\n").encode()


def _create_exclusive(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise CorpusError(f"{path} exists with different content; refusing to overwrite")
        return False
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return _create_exclusive(path, data)
    finally:
        os.unlink(tmp)
    return True


def write(corpus: Corpus, directory: Path) -> str:
    jsonl_path, manifest_path = _paths(directory)
    jsonl = _jsonl(corpus.items)
    if jsonl_path.exists():
        existing = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
        if existing != corpus.digest:
            raise CorpusError(f"{jsonl_path} has digest {existing}, not {corpus.digest}; refusing to overwrite")
    Path(directory).mkdir(parents=True, exist_ok=True)
    created = _create_exclusive(jsonl_path, jsonl)
    created = _create_exclusive(manifest_path, _manifest(corpus)) or created
    return "created" if created else "unchanged"


def load(directory: Path) -> Corpus:
    jsonl_path, manifest_path = _paths(directory)
    for path in (jsonl_path, manifest_path):
        if not path.is_file():
            raise CorpusError(f"missing corpus file {path}")
    raw = jsonl_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        manifest = json.loads(manifest_path.read_bytes())
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        items = tuple(CorpusItem(item=r["item"], template=r["template"], text=r["text"],
                                 token_ids=tuple(r["token_ids"]), sha256=r["sha256"]) for r in rows)
    except (ValueError, KeyError, TypeError) as e:
        raise CorpusError(f"cannot parse corpus files: {e}") from e
    if manifest.get("digest") != digest:
        raise CorpusError(f"jsonl digest {digest} does not match manifest digest {manifest.get('digest')}")
    if (manifest.get("version"), manifest.get("model"), manifest.get("size")) != (VERSION, MODEL, SIZE):
        raise CorpusError("manifest version, model or size does not match this generator")
    for r in rows:
        if r["n_tokens"] != len(r["token_ids"]):
            raise CorpusError(f"item {r['item']}: n_tokens does not match token_ids")
    _check_items(items)
    counted = census(items)
    if manifest.get("census") != counted:
        raise CorpusError("manifest census does not match the items")
    return Corpus(
        version=manifest["version"], model=manifest["model"], tokenizer_revision=manifest["tokenizer_revision"],
        chat_template_sha256=manifest["chat_template_sha256"],
        transformers_version=manifest["transformers_version"], generator_sha256=manifest["generator_sha256"],
        items=items, digest=digest, census=counted,
    )
