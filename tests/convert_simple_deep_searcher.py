"""Convert simple-deep-searcher JSONL to normalized Search-R1 messages.

Default feedback is the doc_gen summary actually observed by the model.
--feedback raw substitutes original pages, changing the observation distribution.
No gold answers are substituted for generated answers; malformed rows fail loudly.
"""

import argparse
import json
from pathlib import Path
import re
import runpy

PROTOCOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "search_comp/data/trajectory.py"))
SEARCH = re.compile(r"<\|begin_search_query\|>(.*?)<\|end_search_query\|>", re.S)
RESULT = re.compile(r"\s*<\|begin_search_result\|>(.*?)<\|end_search_result\|>\s*", re.S)


def clean_thinking(text):
    return re.sub(r"</?(?:think|thinking)>", "", text).strip()


def split_answer(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        raise ValueError("Missing generated boxed answer")
    begin = start + len("\\boxed{")
    depth = 1
    for end in range(begin, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            if text[end + 1:].strip():
                raise ValueError("Unexpected text after boxed answer")
            return clean_thinking(text[:start]), text[begin:end].strip()
    raise ValueError("Unbalanced boxed answer")


def raw_documents(text):
    parts = re.split(r"\*\*Web Page \d+:\*\*\s*", text)
    if parts[0].strip() or len(parts) == 1:
        raise ValueError("Unrecognized raw page format")
    docs = []
    for part in parts[1:]:
        page = json.loads(part)
        docs.append({"title": page.get("title", ""), "text": page.get("context") or page.get("snippet", "")})
    return docs


def convert(row, feedback="summary"):
    messages = [
        {"role": "system", "content": PROTOCOL["BASE_SYSTEM_PROMPT"]},
        {"role": "user", "content": PROTOCOL["SEARCH_INSTRUCTION"] + " Question: " + row["question"]},
    ]
    steps = row["output"]
    index = 0
    searches = 0
    while index < len(steps):
        gen = steps[index].get("gen")
        if not isinstance(gen, str):
            raise ValueError(f"Expected generation at step {index}")
        matches = list(SEARCH.finditer(gen))
        if matches:
            if len(matches) != 1 or gen[matches[0].end():].strip():
                raise ValueError("Expected exactly one terminal search")
            match = matches[0]
            query = match.group(1).strip()
            if steps[index + 1].get("search_query", "").strip() != query:
                raise ValueError("Search query does not match recorded query")
            raw = steps[index + 2].get("doc")
            summary = steps[index + 3].get("doc_gen")
            result = RESULT.fullmatch(summary or "")
            if result is None or not raw:
                raise ValueError("Missing recorded search feedback")
            docs = (raw_documents(raw) if feedback == "raw" else
                    [{"title": "Search result summary", "text": result.group(1).strip()}])
            information = PROTOCOL["format_document_blocks"](docs)
            if not information:
                raise ValueError("Empty search feedback")
            thinking = clean_thinking(gen[:match.start()])
            messages.append({"role": "assistant", "content": f"<thinking>\n{thinking}\n</thinking>\n<search>{query}</search>"})
            messages.append({"role": "user", "content": f"<information>{information}</information>"})
            searches += 1
            index += 4
        else:
            if index != len(steps) - 1:
                raise ValueError("Final answer must be the last step")
            thinking, answer = split_answer(gen)
            if not answer:
                raise ValueError("Empty generated answer")
            messages.append({"role": "assistant", "content": f"<thinking>\n{thinking}\n</thinking>\n<answer>{answer}</answer>"})
            index += 1
    if not messages[-1]["content"].endswith("</answer>"):
        raise ValueError("Missing final answer")
    return {"messages": messages}, searches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feedback", choices=["summary", "raw"], default="summary")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Output must be a new file distinct from input")
    # Validate all rows before creating the output.
    converted = []
    searches = 0
    with args.input.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                sample, count = convert(json.loads(line), args.feedback)
            except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                raise ValueError(f"Line {number}: {exc}") from exc
            converted.append(sample)
            searches += count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        for sample in converted:
            sink.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(converted), "searches": searches, "feedback": args.feedback, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
