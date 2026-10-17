"""Stream OpenSeeker token statistics using cached local Qwen tokenizers.

Native lengths use each original role/content wrapped in ChatML.
Aligned lengths use SearchR1 prompt and encoding rules; tool responses become
information bodies without altering their content. This is a statistical
conversion, not a conversion of the tool protocol into executable Search-R1 data.
"""
import os
os.environ.setdefault('RAYON_NUM_THREADS', '4')
import argparse
import multiprocessing as mp
import json
import sys
import time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from transformers import AutoTokenizer
from search_comp.data.trajectory import SYSTEM_PROMPT, INFO_PREFIX, INFO_SUFFIX

MODELS = ['Qwen/Qwen3.5-2B', 'Qwen/Qwen3-4B-Instruct-2507']


def summary(values):
    a = np.asarray(values, dtype=np.int64)
    return dict(samples=len(values), total_tokens=int(a.sum()), mean_tokens=float(a.mean()),
        median_tokens=float(np.median(a)), min_tokens=int(a.min()), max_tokens=int(a.max()),
        p90_tokens=float(np.percentile(a, 90)), p95_tokens=float(np.percentile(a, 95)),
        over_4096=int((a > 4096).sum()), over_8192=int((a > 8192).sum()),
        over_16384=int((a > 16384).sum()), over_32768=int((a > 32768).sum()))


def initialize_worker():
    global tokenizers, fixed
    tokenizers = {name: AutoTokenizer.from_pretrained(name, local_files_only=True).backend_tokenizer for name in MODELS}
    fixed = {}
    for name, tokenizer in tokenizers.items():
        texts = [f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n',
                 '<|im_start|>assistant\n', INFO_PREFIX, INFO_SUFFIX]
        fixed[name] = [len(e.ids) for e in tokenizer.encode_batch(texts, add_special_tokens=False)]


def measure_row(line):
    roles = Counter()
    row_stats = {}
    obj = json.loads(line); messages = obj['trajectory']
    if not messages or messages[0]['role'] != 'system':
        raise ValueError('Unexpected initial messages in input trajectory')
    if len(messages) < 2 or messages[1]['role'] != 'user':
        raise ValueError('Missing initial user in input trajectory')
    native_texts = []; contents = []; kinds = []
    seen_assistant = False
    for index, msg in enumerate(messages):
        role = msg['role']; content = msg['content']
        if not isinstance(content, str):
            raise ValueError('Non-string content in input trajectory')
        roles[role] += 1
        native_texts.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
        contents.append(content)
        if role == 'assistant':
            kinds.append('assistant'); seen_assistant = True
        elif seen_assistant and role in ('user', 'tool'):
            kinds.append('document')
        elif index < 2:
            kinds.append('prompt')
        else:
            raise ValueError(f'Unexpected role {role} in input trajectory')
    question_text = f"<|im_start|>user\n{obj['question'].strip()}<|im_end|>\n"
    for name, tokenizer in tokenizers.items():
        encoded = tokenizer.encode_batch(native_texts + contents + [question_text], add_special_tokens=False)
        n = len(messages); lengths = [len(e.ids) for e in encoded]
        assistant = sum(lengths[n+i] for i, kind in enumerate(kinds) if kind == 'assistant')
        documents = sum(lengths[n+i] for i, kind in enumerate(kinds) if kind == 'document')
        document_count = kinds.count('document')
        system_len, assistant_prefix, info_prefix, info_suffix = fixed[name]
        aligned = (system_len + lengths[-1] + (assistant_prefix if seen_assistant else 0)
                   + assistant + documents + document_count*(info_prefix+info_suffix))
        row_stats[name] = dict(native=sum(lengths[:n]), aligned=aligned,
            assistant=assistant, documents=documents, raw=sum(lengths[n:2*n]))
    return row_stats, dict(roles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('outputs/data/searchr1/OpenSeeker-v1-Data-trajectories.jsonl'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    output = args.input.with_name(args.input.stem + '-token-stats.json')
    measurements = {name: dict(native=[], aligned=[], assistant=[], documents=[], raw=[]) for name in MODELS}
    roles = Counter(); rows = 0; start = time.time()
    with args.input.open() as source, mp.get_context('spawn').Pool(
            processes=args.workers, initializer=initialize_worker) as pool:
        lines = (line for line in source if line.strip())
        for row_stats, row_roles in pool.imap(measure_row, lines, chunksize=4):
            for name, values in row_stats.items():
                for key, value in values.items():
                    measurements[name][key].append(value)
            roles.update(row_roles)
            rows += 1
            if rows % 500 == 0:
                print(f'{rows} trajectories; elapsed {time.time()-start:.1f}s', flush=True)
    result = dict(data_path=str(args.input.resolve()), source_rows=rows,
        source_bytes=args.input.stat().st_size, message_roles=dict(roles),
        method='Original role/content messages; each message separately encoded as <|im_start|>{role}\\n{content}<|im_end|>\\n; full untruncated sequences; add_special_tokens=False',
        excluded='JSON metadata, standalone answer field, correctness and declared tool-call counts; answers already inside assistant messages remain included',
        aligned_method='SearchR1SFTDataset encoding rules: canonical system, stripped question in initial ChatML user, one initial assistant prefix, assistant content tokens, information prefix/body/suffix for each later user/tool response',
        alignment_note='Original system tool definitions are replaced by the SearchR1 system prompt only for aligned statistics. Assistant output and tool-response bodies (including tool_response tags) are preserved; no tool-call rewriting or document normalization.',
        tokenizers={}, aligned_searchr1_statistics={})
    for name, stats in measurements.items():
        extra = dict(mean_assistant_tokens=float(np.mean(stats['assistant'])),
                     mean_tool_response_tokens=float(np.mean(stats['documents'])))
        result['tokenizers'][name] = dict(**summary(stats['native']), **extra,
            mean_content_tokens=float(np.mean(stats['raw'])))
        result['aligned_searchr1_statistics'][name] = dict(**summary(stats['aligned']), **extra)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
