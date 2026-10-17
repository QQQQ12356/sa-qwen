"""Persist raw trajectory lengths and lengths aligned with SearchR1SFTDataset.

Run from the repository root: python scripts/stats_search_trajectories.py
No text normalization or truncation is applied to outputs or retrieved documents.
"""
import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from transformers import AutoTokenizer
from search_comp.data.searchr1_dataset import SearchR1SFTDataset, align_to_eval_prompt
from search_comp.data.trajectory import SEARCH_INSTRUCTION, format_information_block

SOURCE = Path('outputs/data/searchr1/search-trajectories.jsonl')
BASELINE_STATS = Path('outputs/data/searchr1/qwen3-4b-instruct-sft-normalized-token-stats.json')
OUTPUT = SOURCE.with_name('search-trajectories-token-stats.json')
MODELS = ['Qwen/Qwen3.5-2B', 'Qwen/Qwen3-4B-Instruct-2507']
ORIGINAL_REPORT = dict(
    tokenizer='Qwen/Qwen3.5-2B', implementation='tokenizers.Tokenizer.from_file(tokenizer.json)',
    source_rows=11452, samples=20457, total_tokens=19742487,
    mean_tokens=965.0724446399765, mean_assistant_tokens=286.5692916849978,
    median_tokens=861, min_tokens=179, max_tokens=5429)


def summarize(lengths, assistant, documents):
    values = np.asarray(lengths)
    return dict(samples=len(lengths), total_tokens=int(values.sum()),
                mean_tokens=float(values.mean()), median_tokens=float(np.median(values)),
                min_tokens=int(values.min()), max_tokens=int(values.max()),
                p90_tokens=float(np.percentile(values, 90)),
                p95_tokens=float(np.percentile(values, 95)),
                mean_assistant_tokens=float(np.mean(assistant)),
                mean_document_tokens=float(np.mean(documents)),
                over_4096=int((values > 4096).sum()), over_8192=int((values > 8192).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-baseline', action='store_true',
                        help='Recompute the baseline file referenced by its saved statistics')
    args = parser.parse_args()
    baseline = json.loads(BASELINE_STATS.read_text())
    result = dict(data_path=str(SOURCE.resolve()),
                  method=baseline['method'], source_rows=0,
                  conversion='Each nested trajectory becomes one sample; system/user prompt aligned; generation/final_generation become assistant messages; search.results becomes an information block. Event timestamps and search.query metadata excluded; document/output text unchanged.',
                  tokenizers={},
                  original_report=ORIGINAL_REPORT,
                  previous_raw_statistics=dict(method='Concatenated event content/search.results; no ChatML, prompt alignment or information wrapper; add_special_tokens=False',
                    implementation='transformers.AutoTokenizer; recalculated raw lengths can differ slightly from original low-level Tokenizer report',
                    tokenizers={}),
                  comparison=dict(previous_method_consistent=False,
                    aligned_method_consistent=True, baseline_stats_path=str(BASELINE_STATS),
                    baseline_recomputed=False,
                    note='Same SearchR1SFTDataset encoding and summary definitions; document contents are not normalized, and trajectory <think> tags are preserved.'))
    tokenizers = {name: AutoTokenizer.from_pretrained(name, local_files_only=True) for name in MODELS}
    datasets = {}
    stats = {}
    raw = {}
    for name, tokenizer in tokenizers.items():
        ds = SearchR1SFTDataset.__new__(SearchR1SFTDataset)
        ds.tokenizer = tokenizer; ds.doc_region_split = False; ds.loss_segments = None
        datasets[name] = ds
        stats[name] = ([], [], [])
        raw[name] = ([], [], [])
    event_types = {}
    with SOURCE.open() as f:
        for row_index, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line); result['source_rows'] += 1
            trajectories = obj['trajectories']
            if isinstance(trajectories, str):
                trajectories = json.loads(trajectories)
            for trajectory in trajectories:
                messages = [dict(role='system', content='You are a helpful and harmless assistant.'),
                            dict(role='user', content=SEARCH_INSTRUCTION + ' Question: ' + obj['prompt'])]
                parts = []; outputs = []; documents = []
                for event in trajectory:
                    kind = event['type']; event_types[kind] = event_types.get(kind, 0) + 1
                    if kind == 'prompt':
                        parts.append(event['content'])
                    elif kind in ('generation', 'final_generation'):
                        content = event['content']; parts.append(content); outputs.append(content)
                        messages.append(dict(role='assistant', content=content))
                    elif kind == 'search':
                        content = event['results']; parts.append(content); documents.append(content)
                        messages.append(dict(role='user', content=format_information_block(content)))
                    else:
                        raise ValueError(f'Unknown event type: {kind}')
                sample = dict(messages=align_to_eval_prompt(messages))
                for name, ds in datasets.items():
                    encoded = ds._tokenize(sample)
                    stats[name][0].append(len(encoded['input_ids']))
                    stats[name][1].append(sum(t != -100 for t in encoded['labels']))
                    stats[name][2].append(sum(end-start for start, end in encoded['regions']))
                    tokenizer = tokenizers[name]
                    for dest, text in zip(raw[name], (''.join(parts), ''.join(outputs), ''.join(documents))):
                        dest.append(len(tokenizer(text, add_special_tokens=False).input_ids))
            if row_index % 1000 == 0:
                print(f'Processed {row_index} rows', flush=True)
    result['event_types'] = event_types
    for name in MODELS:
        result['tokenizers'][name] = summarize(*stats[name])
        result['previous_raw_statistics']['tokenizers'][name] = summarize(*raw[name])
        result['comparison'].setdefault('mean_tokens', {})[name] = dict(
            trajectories_aligned=result['tokenizers'][name]['mean_tokens'],
            sft_baseline=baseline['tokenizers'][name]['mean_tokens'])
    if args.verify_baseline:
        verification = {}
        for name, tokenizer in tokenizers.items():
            ds = SearchR1SFTDataset(baseline['data_path'], tokenizer)
            lengths, assistant, documents = [], [], []
            for index in range(len(ds)):
                item = ds[index]
                lengths.append(len(item['input_ids']))
                assistant.append(sum(t != -100 for t in item['labels']))
                documents.append(sum(end-start for start, end in item['regions']))
            measured = summarize(lengths, assistant, documents)
            verification[name] = dict(measured=measured,
                matches_saved=all(measured[k] == v for k, v in baseline['tokenizers'][name].items()))
        result['comparison']['baseline_recomputed'] = True
        result['comparison']['baseline_verification'] = verification
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
