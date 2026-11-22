import os

os.environ["HF_HUB_OFFLINE"] = "1"

import pytest

from e02.corpus import CorpusError, CorpusItem, build, census, load, token_sha256, write


@pytest.fixture(scope="module")
def corpus():
    return build()


def test_build_is_deterministic(corpus):
    assert build().digest == corpus.digest


def test_items_are_distinct_and_in_range(corpus):
    assert len(corpus.items) == 100
    assert [it.item for it in corpus.items] == list(range(100))
    assert len({it.text for it in corpus.items}) == 100
    assert all(960 <= len(it.token_ids) <= 1088 for it in corpus.items)
    assert all(it.sha256 == token_sha256(it.token_ids) for it in corpus.items)


def test_openings_are_problem_specific(corpus):
    assert len({it.text[:16] for it in corpus.items}) == 100
    assert all(it.text[0].isdigit() for it in corpus.items)


def test_rendered_with_qwen_chat_template(corpus):
    first = corpus.items[0].token_ids
    assert first[:3] == (151644, 872, 198)
    assert first[-5:] == (151645, 198, 151644, 77091, 198)
    assert corpus.tokenizer_revision == "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"


def test_little_full_block_sharing(corpus):
    assert corpus.census["pairs"] == 4950
    assert corpus.census["pairs_with_full_block_shared"] == 0


def test_write_then_load_round_trips(corpus, tmp_path):
    assert write(corpus, tmp_path) == "created"
    assert write(corpus, tmp_path) == "unchanged"
    loaded = load(tmp_path)
    assert loaded.items == corpus.items
    assert loaded.digest == corpus.digest
    assert loaded.census == corpus.census
    assert loaded.tokenizer_revision == corpus.tokenizer_revision


def test_load_rejects_flipped_byte(corpus, tmp_path):
    write(corpus, tmp_path)
    path = tmp_path / "corpus-v1.jsonl"
    data = bytearray(path.read_bytes())
    data[200] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(CorpusError, match="digest"):
        load(tmp_path)


def test_load_rejects_missing_manifest(corpus, tmp_path):
    write(corpus, tmp_path)
    (tmp_path / "corpus-v1.manifest.json").unlink()
    with pytest.raises(CorpusError, match="missing"):
        load(tmp_path)


def test_write_refuses_to_overwrite_different_corpus(corpus, tmp_path):
    path = tmp_path / "corpus-v1.jsonl"
    path.write_bytes(b"{}\n")
    with pytest.raises(CorpusError, match="refusing to overwrite"):
        write(corpus, tmp_path)
    assert path.read_bytes() == b"{}\n"
    assert not (tmp_path / "corpus-v1.manifest.json").exists()


def _item(i, ids):
    return CorpusItem(item=i, template=0, text=str(i), token_ids=tuple(ids), sha256=token_sha256(ids))


def test_census_counts_shared_prefixes():
    shared = list(range(1, 21))
    items = [
        _item(0, shared + [100]),
        _item(1, shared + [200, 201]),
        _item(2, [1, 2, 3, 4, 5, 300]),
    ]
    assert census(items) == {
        "pairs": 3,
        "max_common_prefix_tokens": 20,
        "common_prefix_hist": {"5": 2, "20": 1},
        "max_full_block_shared": 1,
        "pairs_with_full_block_shared": 1,
        "length_min": 6,
        "length_max": 22,
        "length_mean": 49 / 3,
        "length_median": 21,
    }
