import ast
import math
import re
from functools import partial

import torch
from torch.utils.data import DataLoader

from data.local_sandbox import local_execute
from data.sampler import InfiniteSampler
from datasets import load_dataset


def extract_code(completion, language="python"):
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    return matches[0] if matches else ""


def get_code_format_reward(language="python"):
    """Return a function that scores format compliance for code responses."""
    pattern = re.compile(
        rf"^(?:(?!```)[\s\S])*?```{language}\n(?:(?!```)[\s\S])*?```\n?$",
        re.DOTALL,
    )

    def code_format_reward(responses):
        rewards = []
        for content in responses:
            if not pattern.fullmatch(content):
                rewards.append(0.0)
                continue
            code_blocks = re.findall(rf"```{language}\n(.*?)```", content, re.DOTALL)
            if not code_blocks:
                rewards.append(0.0)
                continue
            code = code_blocks[0].strip()
            if language == "python":
                try:
                    ast.parse(code)
                    rewards.append(1.0)
                except Exception:
                    rewards.append(0.5)
            else:
                rewards.append(1.0)
        return rewards

    return code_format_reward


def collate_fn_code(batch):
    problems, test_cases, sample_ids = [], [], []
    for item in batch:
        question = item["question"]
        tests = item["test_cases"]
        if "\n```\n" not in question and tests:
            question = question + "\nTest cases: " + tests[0]
        problems.append(question)
        test_cases.append(tests)
        if "__sample_id__" in item:
            sample_ids.append(int(item["__sample_id__"]))
    result = {"problems": problems, "test_cases": test_cases}
    if len(sample_ids) == len(batch):
        result["sample_ids"] = sample_ids
    return result


def reward_code(
    batch,
    responses,
    num_generations,
    device,
    format_reward_alpha=1.0,
    return_components=False,
):
    """Compute the code reward and optionally expose its components."""
    if (
        isinstance(format_reward_alpha, bool)
        or not isinstance(format_reward_alpha, (int, float))
        or not math.isfinite(float(format_reward_alpha))
        or format_reward_alpha < 0
    ):
        raise ValueError("format_reward_alpha must be finite and non-negative")
    test_cases_expanded = batch["test_cases"] * num_generations
    format_rewards = get_code_format_reward()(responses)

    pairs = []
    valid_indices = []
    for i, (fmt, response) in enumerate(zip(format_rewards, responses)):
        if fmt < 1.0:
            continue
        pairs.append((extract_code(response), test_cases_expanded[i]))
        valid_indices.append(i)

    pass_rates = local_execute(pairs) if pairs else []

    format_tensor = torch.tensor(
        format_rewards, device=device, dtype=torch.float32
    )
    pass_tensor = torch.zeros(
        len(responses), device=device, dtype=torch.float32
    )
    for idx, rate in zip(valid_indices, pass_rates):
        pass_tensor[idx] = float(rate)

    rewards = pass_tensor + format_tensor * float(format_reward_alpha)
    if return_components:
        return rewards, {
            "format_reward": format_tensor,
            "pass_rate": pass_tensor,
            "format_valid": (format_tensor >= 1.0).float(),
            "semantic_pass": (pass_tensor > 0.0).float(),
        }
    return rewards


def load_code_dataset_and_reward(
    local_path,
    batch_size,
    num_workers=4,
    seed=112,
    start_position=0,
    format_reward_alpha=1.0,
):
    if local_path.endswith((".jsonl", ".json")):
        ds = load_dataset("json", data_files=local_path, split="train")
    else:
        ds = load_dataset(local_path, split="train")

    ds = ds.add_column("__sample_id__", list(range(len(ds))))
    ds = ds.with_format("torch")
    ds = ds.shuffle(seed=seed)

    sampler = InfiniteSampler(ds, start_position=start_position)
    dataloader = DataLoader(
        ds,
        collate_fn=collate_fn_code,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return dataloader, partial(
        reward_code, format_reward_alpha=format_reward_alpha
    )
