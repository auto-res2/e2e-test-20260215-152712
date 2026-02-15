import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset

NUM_RE = re.compile(r"(-?\d+\.?\d*)")


def extract_last_number(text: str) -> Optional[str]:
    if text is None:
        return None
    matches = NUM_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def extract_yesno(text: str) -> Optional[str]:
    if text is None:
        return None
    t = text.strip().lower()
    if "yes" in t and "no" in t:
        return "yes" if t.rfind("yes") > t.rfind("no") else "no"
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"
    return None


def extract_identity(text: str) -> Optional[str]:
    return text.strip() if text is not None else None


def normalize_number(ans: Optional[str]) -> Optional[float]:
    if ans is None:
        return None
    ans = ans.replace(",", "").strip().strip(".")
    try:
        return float(Decimal(ans))
    except (InvalidOperation, ValueError):
        return None


def normalize_string(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


@dataclass
class AnswerProcessor:
    name: str
    extract_fn: callable
    normalize_fn: callable

    def extract(self, text: str) -> Optional[str]:
        return self.extract_fn(text)

    def normalize(self, ans: Optional[str]):
        return self.normalize_fn(ans)

    def is_correct(self, pred: Optional[str], gold: Optional[str]) -> bool:
        pn = self.normalize(pred)
        gn = self.normalize(gold)
        if pn is None or gn is None:
            return False
        if isinstance(pn, float) and isinstance(gn, float):
            if abs(pn - gn) <= 1e-4 or (abs(gn) > 0 and abs(pn - gn) / abs(gn) <= 1e-4):
                return True
            return False
        return pn == gn


def compute_correct_vector(processor: AnswerProcessor, preds: List[Optional[str]], golds: List[Optional[str]]):
    vec = []
    for p, g in zip(preds, golds):
        vec.append(1 if processor.is_correct(p, g) else 0)
    return np.array(vec, dtype=np.int32)


def _load_hf_split(name: str, config: Optional[str], split: str) -> List[Dict]:
    return load_dataset(name, config, split=split, cache_dir=".cache/")


def _load_gsm8k(cfg) -> Dict[str, List[Dict]]:
    if "splits" not in cfg or not cfg.splits:
        raise ValueError("Dataset splits are required for GSM8K.")
    demo_pool = _load_hf_split(cfg.name, cfg.config, cfg.splits.demo_pool)
    probe = _load_hf_split(cfg.name, cfg.config, cfg.splits.probe)
    test = _load_hf_split(cfg.name, cfg.config, cfg.splits.test)

    processor = AnswerProcessor("gsm8k", extract_last_number, normalize_number)

    return {
        "demo_pool": [{"question": ex["question"], "answer": extract_last_number(ex["answer"])} for ex in demo_pool],
        "probe": [{"question": ex["question"], "answer": extract_last_number(ex["answer"])} for ex in probe],
        "test": [{"question": ex["question"], "answer": extract_last_number(ex["answer"])} for ex in test],
        "processor": processor,
    }


def _load_strategyqa(cfg) -> Dict[str, List[Dict]]:
    if "splits" not in cfg or not cfg.splits:
        raise ValueError("Dataset splits are required for StrategyQA.")
    demo_pool = _load_hf_split(cfg.name, cfg.config, cfg.splits.demo_pool)
    probe = _load_hf_split(cfg.name, cfg.config, cfg.splits.probe)
    test = _load_hf_split(cfg.name, cfg.config, cfg.splits.test)

    def map_ex(ex):
        answer = ex.get("answer")
        if isinstance(answer, bool):
            answer = "yes" if answer else "no"
        elif isinstance(answer, str):
            answer = extract_yesno(answer)
        return {"question": ex["question"], "answer": answer}

    processor = AnswerProcessor("strategyqa", extract_yesno, normalize_string)

    return {
        "demo_pool": [map_ex(ex) for ex in demo_pool],
        "probe": [map_ex(ex) for ex in probe],
        "test": [map_ex(ex) for ex in test],
        "processor": processor,
    }


def _generate_last_letter(n_items: int = 200, seed: int = 0) -> List[Dict]:
    random.seed(seed)
    words = ["tiger", "apple", "banana", "cherry", "delta", "omega", "python", "zephyr"]
    items = []
    for _ in range(n_items):
        sample = random.sample(words, k=3)
        question = "Take the last letters of these words and concatenate them: " + ", ".join(sample)
        answer = "".join(w[-1] for w in sample)
        items.append({"question": question, "answer": answer})
    return items


def _load_synthetic_last_letter(cfg) -> Dict[str, List[Dict]]:
    demo_pool = _generate_last_letter(256, seed=0)
    probe = _generate_last_letter(64, seed=1)
    test = _generate_last_letter(200, seed=2)
    processor = AnswerProcessor("last_letter", extract_identity, normalize_string)
    return {"demo_pool": demo_pool, "probe": probe, "test": test, "processor": processor}


def load_dataset_splits(cfg) -> Dict[str, List[Dict]]:
    name = cfg.name.lower()
    if "gsm8k" in name:
        return _load_gsm8k(cfg)
    if "strategyqa" in name:
        return _load_strategyqa(cfg)
    if "last_letter" in name or "synthetic" in name:
        return _load_synthetic_last_letter(cfg)
    raise ValueError(f"Unsupported dataset: {cfg.name}")


def build_demo_prompt(demos_text: str) -> str:
    return demos_text.strip()
