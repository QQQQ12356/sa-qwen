from types import SimpleNamespace
from unittest.mock import Mock

import torch
import pytest

from search_comp.data.retrieval import format_docs_as_reference
from search_comp.data.trajectory import INFO_PREFIX, INFO_SUFFIX, build_search_chat_prompt
from search_comp.evaluation.beacon_interactive_eval import run_beacon_agent


class _CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(character) for character in text])

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


@pytest.mark.parametrize("question_memory_v1", [False, True])
def test_agent_reuses_cache_only_for_new_retrievals_and_resets_between_questions(question_memory_v1):
    tokenizer = _CharacterTokenizer()
    model = Mock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.beacon_config.beacon_ratio = 2
    model.beacon_config.beacon_question_memory_v1 = question_memory_v1
    outputs = ["<search>first</search>", "<search>second</search>", "<answer>done</answer>"]
    model.beacon_generate.side_effect = [
        torch.tensor([tokenizer(text).input_ids]) for text in outputs * 2
    ]
    retriever = Mock()
    retrieved = [{"title": "Result", "text": "Retrieved content"}]
    retriever.retrieve.return_value = retrieved

    first_result = run_beacon_agent(model, tokenizer, retriever, "Question", max_turns=3)
    second_result = run_beacon_agent(model, tokenizer, retriever, "Question", max_turns=3)

    docs = format_docs_as_reference(retrieved)
    expected_addition = INFO_PREFIX + docs + INFO_SUFFIX
    for call_index, call in enumerate(model.beacon_generate.call_args_list):
        arguments = call.kwargs
        if question_memory_v1:
            assert tokenizer.decode(arguments["question_input_ids"][0].tolist()) == "Question?"
        else:
            assert "question_input_ids" not in arguments
        actual_input = tokenizer.decode(arguments["input_ids"][0].tolist())
        assert arguments["attention_mask"].shape == arguments["input_ids"].shape
        if call_index % 3 == 0:
            assert arguments["reuse_cache"] is False
            assert arguments["regions"] == []
            assert actual_input == build_search_chat_prompt("Question?", add_generation_prompt=True)
        else:
            assert arguments["reuse_cache"] is True
            assert arguments["regions"] == [(len(INFO_PREFIX), len(INFO_PREFIX) + len(docs))]
            assert actual_input == expected_addition

    expected_output = outputs[0] + expected_addition + outputs[1] + expected_addition + outputs[2]
    assert first_result == second_result
    assert first_result["prediction"] == "done"
    assert first_result["turns"] == 2
    assert first_result["output"] == expected_output
    prefix_length = len(build_search_chat_prompt("Question?", add_generation_prompt=True))
    context = build_search_chat_prompt("Question?", add_generation_prompt=True) + expected_output
    assert first_result["context_tokens"] == len(context)
    assert first_result["regions"][0][0] == prefix_length + len(outputs[0]) + len(INFO_PREFIX)
    for start, end in first_result["regions"]:
        assert context[start:end] == docs
