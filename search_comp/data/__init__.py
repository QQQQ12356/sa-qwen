"""数据管线：语料构建、BM25 检索、Search-R1 SFT 数据、Dataset/collator。"""

from .retrieval import (
    BM25Retriever,
    build_corpus_from_hotpotqa,
    format_docs_as_reference,
)
from .searchr1_dataset import SearchR1Collator, SearchR1SFTDataset
from .trajectory import (
    BASE_SYSTEM_PROMPT,
    SEARCH_INSTRUCTION,
    SYSTEM_PROMPT,
    build_search_chat_prompt,
    extract_search_query,
    format_document_blocks,
    format_information_block,
    parse_document_blocks,
)

__all__ = [
    "BM25Retriever",
    "build_corpus_from_hotpotqa",
    "format_docs_as_reference",
    "format_document_blocks",
    "parse_document_blocks",
    "SearchR1SFTDataset",
    "SearchR1Collator",
    "SEARCH_INSTRUCTION",
    "BASE_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "build_search_chat_prompt",
    "extract_search_query",
    "format_information_block",
]
