import math
import os
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from hydra import main
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from tqdm import tqdm

from src.model import LLMWrapper
from src.preprocess import AnswerProcessor, build_demo_prompt, compute_correct_vector, load_dataset_splits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_preds(processor: AnswerProcessor, preds: List[Optional[str]], golds: List[Optional[str]]) -> float:
    correct = compute_correct_vector(processor, preds, golds)
    return float(correct.mean()) if len(correct) > 0 else 0.0


def gen_text(
    model: LLMWrapper,
    prompt: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    n: int,
) -> List[str]:
    return model.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        num_return_sequences=n,
    )


def solve_cot(
    model: LLMWrapper,
    processor: AnswerProcessor,
    q: str,
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> Tuple[str, Optional[str]]:
    prompt = f"Q: {q}\nA: Let's think step by step."
    out = gen_text(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        n=1,
    )[0]
    return out, processor.extract(out)


def build_rewrites(
    model: LLMWrapper,
    q: str,
    *,
    r: int,
    max_new_tokens: int,
) -> List[str]:
    prompt = (
        "Create different wordings of the SAME question without changing its meaning. "
        "Include reformatting, variable/name changes, and one irrelevant sentence. "
        "Do NOT solve. Output one rewrite per line.\n\n"
        f"Question: {q}\nRewrites:\n"
    )
    txt = gen_text(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        n=1,
    )[0]
    lines = [l.strip("- ").strip() for l in txt.splitlines() if l.strip()]
    uniq: List[str] = []
    for l in lines:
        if l and l not in uniq:
            uniq.append(l)
        if len(uniq) >= r:
            break
    return uniq


def same_meaning_bidir(model: LLMWrapper, a: str, b: str, *, max_new_tokens: int) -> bool:
    def yn(prompt: str) -> bool:
        out = gen_text(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            n=1,
        )[0].strip().lower()
        return out.startswith("yes")

    p1 = (
        "Answer yes or no. Do these two questions ask for the same thing (same required answer)?\n"
        f"Q1: {a}\nQ2: {b}\nAnswer:"
    )
    p2 = (
        "Answer yes or no. Do these two questions ask for the same thing (same required answer)?\n"
        f"Q1: {b}\nQ2: {a}\nAnswer:"
    )
    return yn(p1) and yn(p2)


def demo_influence(
    model: LLMWrapper,
    processor: AnswerProcessor,
    demo_text: str,
    probe_q: List[str],
    probe_gold: List[Optional[str]],
    *,
    max_new_tokens: int,
    max_items: Optional[int] = None,
    log_wandb: bool = False,
    step_offset: int = 0,
    base_correct: Optional[np.ndarray] = None,
) -> Tuple[List[Optional[str]], int]:
    preds: List[Optional[str]] = []
    for idx, q in enumerate(probe_q):
        if max_items is not None and idx >= max_items:
            break
        if idx == 0:
            assert isinstance(q, str)
        prompt = f"{demo_text}\nQ: {q}\nA: Let's think step by step."
        out = gen_text(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            n=1,
        )[0]
        pred = processor.extract(out)
        preds.append(pred)
        if log_wandb and base_correct is not None:
            cur_correct = compute_correct_vector(processor, preds, probe_gold[: len(preds)])
            acc1 = float(cur_correct.mean()) if len(cur_correct) > 0 else 0.0
            base_subset = base_correct[: len(preds)]
            acc0 = float(base_subset.mean()) if len(base_subset) > 0 else 0.0
            harm = float(((base_subset == 1) & (cur_correct == 0)).mean()) if len(base_subset) > 0 else 0.0
            wandb.log(
                {
                    "probe_acc0": acc0,
                    "probe_acc1": acc1,
                    "probe_delta_accuracy": acc1 - acc0,
                    "probe_harm_rate": harm,
                    "probe_step": idx,
                },
                step=step_offset,
            )
            step_offset += 1
    return preds, step_offset


def score_candidate_mi(
    q: str,
    model: LLMWrapper,
    processor: AnswerProcessor,
    probe_q: List[str],
    probe_gold: List[Optional[str]],
    base_correct: np.ndarray,
    *,
    m: int,
    r: int,
    tau_sc: float,
    tau_morph: float,
    tau_harm: float,
    alpha: float,
    beta: float,
    lam: float,
    delta: float,
    max_new_tokens: int,
    drift_gate: bool,
    trial_mode: bool,
    log_wandb: bool,
    step_offset: int,
) -> Tuple[Optional[Dict], int]:
    prompt = f"Q: {q}\nA: Let's think step by step."
    samples = gen_text(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        n=m,
    )
    answers = [processor.extract(s) for s in samples]
    answers = [a for a in answers if a is not None]
    if not answers:
        return None, step_offset
    counts: Dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    maj = max(counts.items(), key=lambda x: x[1])[0]
    p_sc = counts[maj] / len(answers)
    if p_sc < tau_sc:
        return None, step_offset
    maj_rats = [s for s in samples if processor.extract(s) == maj]
    chosen = min(maj_rats, key=lambda s: len(s.split()))
    demo_text = f"Q: {q}\nA: {chosen}\n"

    rewrites = build_rewrites(model, q, r=r, max_new_tokens=192)
    kept: List[str] = []
    if drift_gate:
        for rq in rewrites:
            if same_meaning_bidir(model, q, rq, max_new_tokens=4):
                kept.append(rq)
    else:
        kept = rewrites
    if not kept:
        return None, step_offset
    ok = 0
    for rq in kept:
        _, a2 = solve_cot(model, processor, rq, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
        ok += int(a2 == maj)
    p_morph = ok / len(kept)
    if p_morph < tau_morph:
        return None, step_offset

    max_probe_items = 2 if trial_mode else None
    with_preds, step_offset = demo_influence(
        model,
        processor,
        demo_text,
        probe_q,
        probe_gold,
        max_new_tokens=max_new_tokens,
        max_items=max_probe_items,
        log_wandb=log_wandb,
        step_offset=step_offset,
        base_correct=base_correct,
    )
    with_correct = compute_correct_vector(processor, with_preds, probe_gold[: len(with_preds)])
    base_subset = base_correct[: len(with_preds)]
    acc0 = float(base_subset.mean()) if len(with_preds) > 0 else 0.0
    acc1 = float(with_correct.mean()) if len(with_preds) > 0 else 0.0
    dacc = acc1 - acc0
    harm = float(((base_subset == 1) & (with_correct == 0)).mean()) if len(with_preds) > 0 else 0.0
    if dacc <= 0.0 or harm > tau_harm:
        return None, step_offset

    score = (p_sc**alpha) * (p_morph**beta) * (1.0 / (1.0 + np.exp(-lam * dacc))) * ((1.0 - harm) ** delta)
    if log_wandb:
        wandb.log(
            {
                "candidate_p_sc": float(p_sc),
                "candidate_p_morph": float(p_morph),
                "candidate_dacc": float(dacc),
                "candidate_harm": float(harm),
                "candidate_score": float(score),
            },
            step=step_offset,
        )
        step_offset += 1

    return (
        {
            "demo": demo_text,
            "score": float(score),
            "p_sc": float(p_sc),
            "p_morph": float(p_morph),
            "acc0": acc0,
            "acc1": acc1,
            "dacc": float(dacc),
            "harm": harm,
            "kept_rewrites": len(kept),
        },
        step_offset,
    )


def score_candidate_iiw(
    q: str,
    model: LLMWrapper,
    processor: AnswerProcessor,
    probe_q: List[str],
    probe_gold: List[Optional[str]],
    base_correct: np.ndarray,
    *,
    m: int,
    r: int,
    tau_sc: float,
    tau_morph: float,
    max_new_tokens: int,
    drift_gate: bool,
    paraphrase_consistency: bool,
    trial_mode: bool,
    log_wandb: bool,
    step_offset: int,
) -> Tuple[Optional[Dict], int]:
    prompt = f"Q: {q}\nA: Let's think step by step."
    samples = gen_text(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        n=m,
    )
    answers = [processor.extract(s) for s in samples]
    answers = [a for a in answers if a is not None]
    if not answers:
        return None, step_offset
    counts: Dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    maj = max(counts.items(), key=lambda x: x[1])[0]
    p_sc = counts[maj] / len(answers)
    if p_sc < tau_sc:
        return None, step_offset
    maj_rats = [s for s in samples if processor.extract(s) == maj]
    chosen = min(maj_rats, key=lambda s: len(s.split()))
    demo_text = f"Q: {q}\nA: {chosen}\n"

    rewrites = build_rewrites(model, q, r=r, max_new_tokens=192)
    kept = rewrites
    if drift_gate:
        kept = [rq for rq in rewrites if same_meaning_bidir(model, q, rq, max_new_tokens=4)]
    if paraphrase_consistency:
        if not kept:
            return None, step_offset
        ok = 0
        for rq in kept:
            _, a2 = solve_cot(model, processor, rq, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
            ok += int(a2 == maj)
        p_morph = ok / len(kept)
        if p_morph < tau_morph:
            return None, step_offset
    else:
        p_morph = 1.0

    max_probe_items = 2 if trial_mode else None
    with_preds, step_offset = demo_influence(
        model,
        processor,
        demo_text,
        probe_q,
        probe_gold,
        max_new_tokens=max_new_tokens,
        max_items=max_probe_items,
        log_wandb=log_wandb,
        step_offset=step_offset,
        base_correct=base_correct,
    )
    with_correct = compute_correct_vector(processor, with_preds, probe_gold[: len(with_preds)])
    acc1 = float(with_correct.mean()) if len(with_preds) > 0 else 0.0
    acc0 = float(base_correct[: len(with_preds)].mean()) if len(with_preds) > 0 else 0.0
    dacc = acc1 - acc0
    harm = float(((base_correct[: len(with_preds)] == 1) & (with_correct == 0)).mean()) if len(with_preds) > 0 else 0.0
    score = p_sc * p_morph * acc1

    if log_wandb:
        wandb.log(
            {
                "candidate_p_sc": float(p_sc),
                "candidate_p_morph": float(p_morph),
                "candidate_dacc": float(dacc),
                "candidate_harm": float(harm),
                "candidate_score": float(score),
            },
            step=step_offset,
        )
        step_offset += 1

    return (
        {
            "demo": demo_text,
            "score": float(score),
            "p_sc": float(p_sc),
            "p_morph": float(p_morph),
            "acc0": acc0,
            "acc1": acc1,
            "dacc": float(dacc),
            "harm": harm,
            "kept_rewrites": len(kept),
        },
        step_offset,
    )


def build_demos(
    questions: List[str],
    model: LLMWrapper,
    processor: AnswerProcessor,
    probe_q: List[str],
    probe_gold: List[Optional[str]],
    *,
    method: str,
    method_settings: Dict,
    max_new_tokens: int,
    trial_mode: bool,
    log_wandb: bool,
) -> Tuple[str, List[Dict]]:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    k = int(method_settings.get("k_clusters", 8))
    t = int(method_settings.get("oversample_t", 1))
    m = int(method_settings.get("self_consistency_m", 5))
    r = int(method_settings.get("rewrite_r", 4))
    tau_sc = float(method_settings.get("tau_sc", 0.6))
    tau_morph = float(method_settings.get("tau_morph", 0.6))
    drift_gate = method_settings.get("drift_gate", "none") != "none"
    paraphrase_consistency = bool(method_settings.get("paraphrase_consistency", True))

    max_candidates_total = None
    if trial_mode:
        k = min(2, k)
        t = min(1, t)
        m = min(2, m)
        r = min(2, r)
        max_candidates_total = 4

    emb = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=os.path.join(get_original_cwd(), ".cache"))
    X = emb.encode(questions, normalize_embeddings=True, show_progress_bar=False)
    km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)

    max_probe_items = 2 if trial_mode else None
    base_preds: List[Optional[str]] = []
    for idx, q in enumerate(probe_q):
        if max_probe_items is not None and idx >= max_probe_items:
            break
        base_preds.append(
            solve_cot(model, processor, q, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)[1]
        )
        if log_wandb:
            base_correct_partial = compute_correct_vector(processor, base_preds, probe_gold[: len(base_preds)])
            wandb.log(
                {
                    "probe_acc0": float(base_correct_partial.mean()) if len(base_correct_partial) > 0 else 0.0,
                    "probe_step": idx,
                },
                step=idx,
            )
    base_correct = compute_correct_vector(processor, base_preds, probe_gold[: len(base_preds)])

    chosen_infos: List[Dict] = []
    step_offset = 1
    seen_candidates = 0
    for c in range(k):
        idxs = np.where(km.labels_ == c)[0]
        if len(idxs) == 0:
            continue
        centroid = km.cluster_centers_[c]
        sims = X[idxs] @ centroid / (np.linalg.norm(centroid) + 1e-9)
        order = idxs[np.argsort(-sims)]
        candidates = order[: min(t, len(order))]

        best = None
        for i in candidates:
            q = questions[int(i)]
            seen_candidates += 1
            if max_candidates_total is not None and seen_candidates > max_candidates_total:
                break
            if "MI-AutoCoT" in method:
                info, step_offset = score_candidate_mi(
                    q,
                    model,
                    processor,
                    probe_q,
                    probe_gold,
                    base_correct,
                    m=m,
                    r=r,
                    tau_sc=tau_sc,
                    tau_morph=tau_morph,
                    tau_harm=float(method_settings.get("tau_harm", 0.2)),
                    alpha=float(method_settings.get("alpha", 1.0)),
                    beta=float(method_settings.get("beta", 1.0)),
                    lam=float(method_settings.get("lambda", 8.0)),
                    delta=float(method_settings.get("delta", 2.0)),
                    max_new_tokens=max_new_tokens,
                    drift_gate=drift_gate,
                    trial_mode=trial_mode,
                    log_wandb=log_wandb,
                    step_offset=step_offset,
                )
            else:
                info, step_offset = score_candidate_iiw(
                    q,
                    model,
                    processor,
                    probe_q,
                    probe_gold,
                    base_correct,
                    m=m,
                    r=r,
                    tau_sc=tau_sc,
                    tau_morph=tau_morph,
                    max_new_tokens=max_new_tokens,
                    drift_gate=drift_gate,
                    paraphrase_consistency=paraphrase_consistency,
                    trial_mode=trial_mode,
                    log_wandb=log_wandb,
                    step_offset=step_offset,
                )
            if info is None:
                continue
            if (best is None) or (info["score"] > best["score"]):
                best = info
        if best is not None:
            chosen_infos.append(best)
        if max_candidates_total is not None and seen_candidates > max_candidates_total:
            break

    chosen_infos.sort(key=lambda d: -d["score"])
    demos_text = "\n".join(d["demo"] for d in chosen_infos[:k])
    return demos_text, chosen_infos


def adjust_for_mode(cfg) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if "optuna" in cfg:
            cfg.optuna.n_trials = 0
            cfg.optuna.enabled = False
        if "training" in cfg:
            cfg.training.epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"


def train_supervised_loop(
    model: LLMWrapper,
    train_pairs: List[Tuple[str, str]],
    *,
    batch_size: int,
    lr: float,
    epochs: int,
    log_wandb: bool,
    max_batches: Optional[int] = None,
) -> None:
    if epochs <= 0:
        return
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr)
    tokenizer = model.tokenizer
    device = model.device
    step = 0
    for epoch in range(epochs):
        random.shuffle(train_pairs)
        for i in range(0, len(train_pairs), batch_size):
            if max_batches is not None and (i // batch_size) >= max_batches:
                break
            batch = train_pairs[i : i + batch_size]
            prompts = [f"Q: {q}\nA:" for q, _ in batch]
            answers = [a for _, a in batch]

            if step == 0:
                assert len(prompts) == len(answers)

            if model.is_encoder_decoder:
                enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                lab = tokenizer(answers, return_tensors="pt", padding=True, truncation=True).to(device)
                labels = lab.input_ids
                if step == 0:
                    assert enc.input_ids.shape[0] == labels.shape[0]
                    assert enc.input_ids.ndim == 2 and labels.ndim == 2
                outputs = model.model(**enc, labels=labels)
            else:
                full_text = [f"{p} {a}" for p, a in zip(prompts, answers)]
                enc = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True).to(device)
                labels = enc.input_ids.clone()
                prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                prompt_lengths = prompt_enc.input_ids.ne(tokenizer.pad_token_id).sum(dim=1)
                for idx, pl in enumerate(prompt_lengths.tolist()):
                    labels[idx, :pl] = -100
                if step == 0:
                    assert enc.input_ids.shape == labels.shape
                    assert labels.ndim == 2
                outputs = model.model(**enc, labels=labels)

            loss = outputs.loss
            if log_wandb:
                wandb.log({"train_loss": float(loss.item()), "epoch": epoch}, step=step)
            loss.backward()

            grads = [p.grad for p in model.model.parameters() if p.requires_grad]
            assert any(g is not None for g in grads), "No gradients found before optimizer.step()"
            assert any((g is not None and torch.any(g != 0)) for g in grads), (
                "All gradients are zero before optimizer.step()"
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1


def run_single(cfg) -> Dict[str, float]:
    set_seed(int(cfg.seed))

    dataset = load_dataset_splits(cfg.dataset)
    processor = dataset["processor"]
    demo_pool = dataset["demo_pool"]
    probe = dataset["probe"]
    test = dataset["test"]

    if cfg.mode == "trial":
        demo_pool = demo_pool[:16]
        probe = probe[:8]
        test = test[:8]

    cache_dir = os.path.join(get_original_cwd(), ".cache")
    model = LLMWrapper(cfg.model, cache_dir=cache_dir)
    assert model.tokenizer.pad_token_id is not None
    assert model.model.config.vocab_size > 0
    out_emb = model.model.get_output_embeddings()
    if out_emb is not None:
        assert out_emb.weight.shape[0] == model.model.config.vocab_size

    demo_q = [ex["question"] for ex in demo_pool]
    probe_q = [ex["question"] for ex in probe]
    probe_gold = [ex["answer"] for ex in probe]
    test_q = [ex["question"] for ex in test]
    test_gold = [ex["answer"] for ex in test]

    assert len(probe_q) == len(probe_gold)
    assert len(test_q) == len(test_gold)

    if len(test_q) > 0:
        assert isinstance(test_q[0], str)

    max_new_tokens = int(cfg.model.max_new_tokens)
    if cfg.mode == "trial":
        max_new_tokens = min(32, max_new_tokens)

    demos_text, meta = build_demos(
        demo_q,
        model,
        processor,
        probe_q,
        probe_gold,
        method=cfg.method,
        method_settings=cfg.method_settings,
        max_new_tokens=max_new_tokens,
        trial_mode=cfg.mode == "trial",
        log_wandb=cfg.wandb.mode != "disabled",
    )

    demo_prompt = build_demo_prompt(demos_text)

    test_preds: List[Optional[str]] = []
    test_correct_flags: List[int] = []
    max_eval_steps = 2 if cfg.mode == "trial" else None
    for idx, q in enumerate(tqdm(test_q, desc="Testing")):
        if max_eval_steps is not None and idx >= max_eval_steps:
            break
        if idx == 0:
            assert isinstance(q, str)
            assert len(test_gold) >= 1
        prompt = f"{demo_prompt}\nQ: {q}\nA: Let's think step by step."
        out = gen_text(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=float(cfg.model.greedy_temperature),
            n=1,
        )[0]
        pred = processor.extract(out)
        test_preds.append(pred)
        is_correct = int(processor.is_correct(pred, test_gold[idx]))
        test_correct_flags.append(is_correct)
        if cfg.wandb.mode != "disabled":
            running_acc = float(np.mean(test_correct_flags)) if test_correct_flags else 0.0
            wandb.log(
                {
                    "test_step": idx,
                    "test_correct": is_correct,
                    "test_accuracy": running_acc,
                },
                step=idx,
            )

    test_accuracy = accuracy_from_preds(processor, test_preds, test_gold[: len(test_preds)])

    probe_demo_preds, _ = demo_influence(
        model,
        processor,
        demo_prompt,
        probe_q,
        probe_gold,
        max_new_tokens=max_new_tokens,
        max_items=max_eval_steps,
        log_wandb=False,
    )
    base_probe_preds: List[Optional[str]] = []
    for idx, q in enumerate(probe_q):
        if max_eval_steps is not None and idx >= max_eval_steps:
            break
        if idx == 0:
            assert isinstance(q, str)
            assert len(probe_gold) >= 1
        base_probe_preds.append(
            solve_cot(model, processor, q, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)[1]
        )
    base_correct = compute_correct_vector(processor, base_probe_preds, probe_gold[: len(base_probe_preds)])
    demo_correct = compute_correct_vector(processor, probe_demo_preds, probe_gold[: len(probe_demo_preds)])
    acc0 = float(base_correct.mean()) if len(base_correct) > 0 else 0.0
    acc1 = float(demo_correct.mean()) if len(demo_correct) > 0 else 0.0
    probe_delta = acc1 - acc0
    harm_rate = float(((base_correct == 1) & (demo_correct == 0)).mean()) if len(base_correct) > 0 else 0.0

    probe_cc = int(((base_correct == 1) & (demo_correct == 1)).sum())
    probe_wc = int(((base_correct == 0) & (demo_correct == 1)).sum())
    probe_cw = int(((base_correct == 1) & (demo_correct == 0)).sum())
    probe_ww = int(((base_correct == 0) & (demo_correct == 0)).sum())

    p_sc_values = [m["p_sc"] for m in meta if "p_sc" in m]
    p_morph_values = [m["p_morph"] for m in meta if "p_morph" in m]

    metrics = {
        "test_accuracy": test_accuracy,
        "probe_harm_rate": harm_rate,
        "probe_delta_accuracy": probe_delta,
        "demo_stability_p_sc": float(np.mean(p_sc_values)) if p_sc_values else 0.0,
        "metamorphic_invariance_p_morph": float(np.mean(p_morph_values)) if p_morph_values else 0.0,
        "demo_selection_yield": len(meta) / float(cfg.method_settings.get("k_clusters", 8)),
        "probe_acc0": acc0,
        "probe_acc1": acc1,
        "probe_cc": probe_cc,
        "probe_wc": probe_wc,
        "probe_cw": probe_cw,
        "probe_ww": probe_ww,
    }

    if cfg.wandb.mode != "disabled":
        wandb.log(metrics, step=max(len(test_correct_flags), 1) + 1)

    return metrics


def run_optuna(cfg) -> Dict[str, float]:
    import optuna

    search_spaces = cfg.optuna.get("search_spaces") if "optuna" in cfg else None
    if not search_spaces:
        return {}

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = deepcopy(cfg)
        trial_cfg.wandb.mode = "disabled"
        for key, spec in search_spaces.items():
            if spec["type"] == "float":
                val = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "int":
                val = trial.suggest_int(key, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                val = trial.suggest_categorical(key, spec["choices"])
            else:
                continue
            trial_cfg.method_settings[key] = val
        metrics = run_single(trial_cfg)
        return metrics.get("test_accuracy", 0.0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(cfg.optuna.n_trials))
    return study.best_params


def run_experiment(cfg) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    adjust_for_mode(cfg)

    original_cwd = get_original_cwd()
    results_dir = cfg.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(original_cwd, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    best_params: Dict[str, float] = {}
    if cfg.optuna.get("enabled", False) and int(cfg.optuna.n_trials) > 0:
        best_params = run_optuna(cfg)
        for k, v in best_params.items():
            cfg.method_settings[k] = v

    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(f"WandB URL: {wandb.run.url}")

    if not cfg.training.inference_only and cfg.training.epochs > 0:
        dataset = load_dataset_splits(cfg.dataset)
        train_pairs = [(ex["question"], ex["answer"]) for ex in dataset["demo_pool"]]
        cache_dir = os.path.join(original_cwd, ".cache")
        model = LLMWrapper(cfg.model, cache_dir=cache_dir)
        assert model.tokenizer.pad_token_id is not None
        assert model.model.config.vocab_size > 0
        max_batches = 2 if cfg.mode == "trial" else None
        train_supervised_loop(
            model,
            train_pairs,
            batch_size=int(cfg.training.batch_size),
            lr=float(cfg.training.learning_rate),
            epochs=int(cfg.training.epochs),
            log_wandb=cfg.wandb.mode != "disabled",
            max_batches=max_batches,
        )

    metrics = run_single(cfg)

    if cfg.wandb.mode != "disabled":
        for k, v in metrics.items():
            wandb.summary[k] = v
        wandb.finish()


@main(config_path="../config", config_name="config", version_base="1.3")
def run(cfg) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    run()
