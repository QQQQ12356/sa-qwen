"""``<information>`` 文档块统一格式的单元测试。

覆盖训练/评测共用的渲染（``Doc N <标题>`` + 去重正文、空行分隔、无 ID/Score 元信息）
与解析（兼容 ``[Document N] (ID:, Score:)``、``Doc N (Title: ...)`` 与统一格式），
以及渲染↔解析的幂等性。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_comp.data.retrieval import format_docs_as_reference
from search_comp.data.trajectory import (
    DOC_SEPARATOR,
    format_document_block,
    format_document_blocks,
    parse_document_blocks,
)

#: 真实 Search-R1 轨迹里的三种形态：标题重复两次 / 标题只在正文开头 / 正文不含标题。
_LEGACY_INFO = (
    "<information>[Document 1] (ID: 1160246, Score: 0.861)\n"
    '"DeLorean time machine"\n'
    "DeLorean time machine The DeLorean time machine is a fictional car.\n"
    "\n"
    "[Document 2] (ID: 18901013, Score: 0.876)\n"
    '"Anil Kumble"\n'
    "Anil Kumble Anil Kumble ( born 17 October 1970) is a former Indian cricketer.\n"
    "\n"
    "[Document 3] (ID: 42, Score: 0.5)\n"
    '"Hill Valley (Back to the Future)"\n'
    "many ways in which the Courthouse building has been redressed.\n"
    "</information>"
)


def _docs(title, body):
    return [{"title": title, "text": f"{title}\n{body}"}]


def test_format_document_block_keeps_body_when_title_appears_once():
    # 正文开头只出现一次标题时原样保留（句子本就完整）。
    block = format_document_block(
        1, "Anil Kumble", "Anil Kumble\nAnil Kumble ( born 17 October 1970) is a cricketer."
    )
    assert block == (
        "Doc 1 Anil Kumble\nAnil Kumble ( born 17 October 1970) is a cricketer."
    )


def test_format_document_block_collapses_consecutive_repeated_title():
    # 标题连续重复两次 -> 折叠成一个（保留标题，且不会把句子切成 "( born ..."）。
    block = format_document_block(
        1, "Anil Kumble", "Anil Kumble\nAnil Kumble Anil Kumble ( born 1970) is a cricketer."
    )
    assert block == "Doc 1 Anil Kumble\nAnil Kumble ( born 1970) is a cricketer."


def test_collapse_repeated_title_handles_more_than_two():
    from search_comp.data.trajectory import collapse_repeated_title

    assert collapse_repeated_title("T", "T T T T rest") == "T rest"
    assert collapse_repeated_title("T", "T T") == "T"
    assert collapse_repeated_title("T", "T rest") == "T rest"
    assert collapse_repeated_title("T", "rest") == "rest"
    # 标题只是更长词的前缀时不算重复。
    assert collapse_repeated_title("Matoma", "Matoma Matomaa rest") == "Matoma Matomaa rest"


def test_format_document_block_never_starts_body_with_punctuation():
    # 折叠的目标：正文开头永远是标题（合法句子开头），不会只剩标点。
    for body in ("T T ( born 1970) x", "T T, rest", "T T T"):
        block = format_document_block(1, "T", f"T\n{body}")
        rendered_body = block.split("\n", 1)[1]
        assert rendered_body.startswith("T")


def test_format_document_block_keeps_body_without_leading_title():
    block = format_document_block(
        1, "Hill Valley (Back to the Future)", "Hill Valley (Back to the Future)\nmany ways ..."
    )
    assert block == "Doc 1 Hill Valley (Back to the Future)\nmany ways ..."


def test_format_document_blocks_body_matches_source_after_collapse():
    # 渲染后的正文 == 语料正文（仅在开头连续重复标题时折叠成一个）。
    from search_comp.data.trajectory import collapse_repeated_title

    docs = parse_document_blocks(_LEGACY_INFO)
    rendered = format_document_blocks(docs)
    for index, doc in enumerate(docs, start=1):
        body = collapse_repeated_title(doc["title"], doc["text"].split("\n", 1)[1])
        assert f"Doc {index} {doc['title']}\n{body}" in rendered


def test_format_document_blocks_skips_empty_docs_and_numbers_continuously():
    blocks = format_document_blocks(
        [
            {"title": "A", "text": "A\nbody a"},
            {"title": "B", "text": "   "},
            {"title": "C", "text": "C\nbody c"},
        ]
    )
    assert blocks == f"Doc 1 A\nbody a{DOC_SEPARATOR}Doc 2 C\nbody c"


def test_format_document_blocks_has_no_retrieval_metadata():
    rendered = format_document_blocks(parse_document_blocks(_LEGACY_INFO))
    for marker in ("[Document", "(ID:", "Score:", "(Title:"):
        assert marker not in rendered
    assert rendered.startswith(
        "Doc 1 DeLorean time machine\n"
        "DeLorean time machine The DeLorean time machine is a fictional car."
    )


def test_format_docs_as_reference_matches_unified_format():
    # 评测侧入口（BM25 检索结果 -> <information> 文本）必须与训练侧同格式。
    retrieved = [
        {"id": "1", "title": "DeLorean time machine",
         "text": "DeLorean time machine\nDeLorean time machine The DeLorean time machine is a car."},
    ]
    assert format_docs_as_reference(retrieved) == format_document_blocks(retrieved)


def test_parse_document_blocks_accepts_legacy_and_unified_formats():
    from_legacy = parse_document_blocks(_LEGACY_INFO)
    assert [d["title"] for d in from_legacy] == [
        "DeLorean time machine", "Anil Kumble", "Hill Valley (Back to the Future)",
    ]
    assert from_legacy[0]["text"] == (
        "DeLorean time machine\nDeLorean time machine The DeLorean time machine is a fictional car."
    )
    # 折叠后的正文在再次解析/渲染时保持不变（标题与正文都稳定）。
    from search_comp.data.trajectory import collapse_repeated_title

    unified = format_document_blocks(from_legacy)
    reparsed = parse_document_blocks(unified)
    assert [d["title"] for d in reparsed] == [d["title"] for d in from_legacy]
    assert [d["text"] for d in reparsed] == [
        f"{d['title']}\n"
        f"{collapse_repeated_title(d['title'], d['text'].split(chr(10), 1)[1])}"
        for d in from_legacy
    ]
    assert format_document_blocks(reparsed) == unified


def test_parse_document_blocks_accepts_titled_legacy_format():
    # 旧版评测渲染（Doc N (Title: ...) <标题>\n<正文>）仍可解析，便于规范化老数据。
    old = "Doc 1 (Title: Chelsea Handler) Chelsea Handler\nSeymour Handler, a used car dealer."
    assert parse_document_blocks(old) == [
        {"id": "Chelsea Handler|||Seymour Handler, a used car dealer.",
         "title": "Chelsea Handler",
         "text": "Chelsea Handler\nSeymour Handler, a used car dealer."}
    ]


def test_parse_document_blocks_handles_empty_and_untagged():
    assert parse_document_blocks("") == []
    assert parse_document_blocks("no document here") == []
    # 不含外层标签时按块内文本解析。
    assert len(parse_document_blocks(
        "[Document 1] (ID: 1, Score: 0.5)\n\"T\"\nT body."
    )) == 1


def test_render_parse_round_trip_is_idempotent():
    once = format_document_blocks(parse_document_blocks(_LEGACY_INFO))
    twice = format_document_blocks(parse_document_blocks(once))
    assert once == twice
