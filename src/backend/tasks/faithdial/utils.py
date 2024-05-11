from typing import List, Union

ValueType = Union[str, List[str]]


def doc_to_text(doc: dict[str, ValueType]) -> str:
    history_str = " ".join([f'[{"Human" if i % 2 == 0 else "Assistant"}] {m}' for i, m in enumerate(doc["history"])])
    doc_text = f'#Knowledge#: {doc["knowledge"]}\n#Dialogue History#: {history_str}\n#Response#: {doc["response"]}\n#Hallucinated#:'
    return doc_text


def doc_to_text_v2(doc: dict[str, ValueType]) -> str:
    history_str = " ".join([f'[{"Human" if i % 2 == 0 else "Assistant"}] {m}' for i, m in enumerate(doc["history"])])
    doc_text = f'#Knowledge#: {doc["knowledge"]}\n#Dialogue History#: {history_str}\n#Response#: {doc["original_response"]}\n#Hallucinated#:'
    return doc_text


def doc_to_target(doc: dict[str, ValueType]) -> str:
    res = "true" if "Hallucination" in doc["BEGIN"] else "false"
    return res
