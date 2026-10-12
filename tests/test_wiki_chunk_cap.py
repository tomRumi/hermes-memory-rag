"""ingest_wiki chunking must respect the embedding context for accented text."""
from hermes_memory_rag import server


def test_long_hungarian_section_split_under_cap_with_heading():
    body = "árvíztűrő tükörfúrógép és a teremtésvédelem " * 300
    text = f"# Cím\n\nbevezető\n\n## Kategóriafa\n{body}"
    pieces = server._wiki_chunks(text, max_chars=2400)
    sec = [p for p in pieces if p.startswith("## Kategóriafa\n")]
    assert len(sec) > 1
    assert all(len(p) <= 2400 for p in pieces)
    assert all(p.startswith("## Kategóriafa\n") for p in sec)


def test_fitting_section_stays_atomic():
    pieces = server._wiki_chunks("intro\n\n## A\nshort body", max_chars=2400)
    assert pieces == ["intro", "## A\nshort body"]


def test_default_cap_below_code_cap():
    assert server.WIKI_CHUNK_CHAR_CAP < server.CHUNK_CHAR_CAP
