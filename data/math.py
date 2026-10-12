# Adapted from https://github.com/maple-research-lab/LLaDOU/blob/main/dataloaders/math.py
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from data.sampler import InfiniteSampler
from utils.evaluation.grader import math_equal
from utils.evaluation.parser import extract_answer, parse_ground_truth


def _load_local_parquet(path: Path, split: str):
    data_file = path if path.is_file() else path / f"{split}-*.parquet"
    return load_dataset(
        "parquet",
        data_files={split: str(data_file)},
        split=split,
    )


def _training_data_path(local_path: str) -> Path:
    path = Path(local_path).expanduser()
    if not path.exists() and (
        path.is_absolute()
        or local_path.startswith(("./", "../", "~", "datasets/"))
        or path.suffix in (".parquet", ".arrow", ".json", ".jsonl")
    ):
        raise FileNotFoundError(f"Training data path does not exist: {local_path}")
    return path


def collate_fn_gsm8k(batch):
    """Collate function for GSM8K dataset."""
    problems = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    result = {"problems": problems, "answers": answers}
    if all('__sample_id__' in item for item in batch):
        result["sample_ids"] = [int(item['__sample_id__']) for item in batch]
    return result


def extract_answer_gsm8k(answer: str):
    """Extract the final answer from GSM8K format (after ####)."""
    return answer.split('####')[-1].strip()


def reward_gsm8k(batch, responses, num_generations, device, return_components=False):
    """
    Compute reward for GSM8K responses.
    
    Args:
        batch: Batch containing ground truth answers
        responses: Model generated responses
        num_generations: Number of generations per problem
        device: Torch device
    
    Returns:
        Tensor of rewards (+1 for correct, -1 for incorrect)
    """
    answers = batch['answers'] * num_generations
    
    ext_ans = [extract_answer_gsm8k(ans) for ans in answers]
    ext_res = [extract_answer(res) for res in responses]
    
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(ext_ans, ext_res)):
        if math_equal(ans, res):
            rewards[i] = 1.0
        else:
            rewards[i] = -1.0
    
    if return_components:
        return rewards, {
            "accuracy_reward": rewards,
            "correct": (rewards > 0).float(),
        }
    return rewards


def load_gsm8k_dataset_and_reward(
    local_path: str = "gsm8k",
    batch_size: int = 1,
    split: str = 'train',
    num_workers: int = 4,
    seed: int = 112,
    start_position: int = 0,
):
    """
    Load GSM8K dataset and return dataloader with reward function.
    
    Args:
        local_path: HuggingFace dataset path
        batch_size: Batch size per GPU
        split: Dataset split to use
        num_workers: Number of dataloader workers
        seed: Random seed for shuffling
        start_position: Absolute position in the global sample stream
    
    Returns:
        Tuple of (dataloader, reward_function)
    """
    dataset_path = _training_data_path(local_path)
    if dataset_path.is_file():
        ds = _load_local_parquet(dataset_path.resolve(), split)
    elif dataset_path.is_dir():
        local_directory = dataset_path / "main"
        if not local_directory.is_dir():
            local_directory = dataset_path
        ds = _load_local_parquet(local_directory.resolve(), split)
    else:
        ds = load_dataset(local_path, "main", split=split)
    ds = ds.add_column('__sample_id__', list(range(len(ds))))
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)

    sampler = InfiniteSampler(ds, start_position=start_position)
    
    dataloader = DataLoader(
        ds,
        collate_fn=collate_fn_gsm8k,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    
    return dataloader, reward_gsm8k


def collate_fn_math(batch):
    """Collate function for MATH dataset."""
    problems = []
    answers = []
    sample_ids = []
    instruct = r"(Please put the final answer in \boxed{} tag, i.e. $\boxed{answer here}$)"
    for item in batch:
        problems.append(item['problem'] + instruct)
        answers.append(item['solution'])
        if '__sample_id__' in item:
            sample_ids.append(int(item['__sample_id__']))
    result = {"problems": problems, "answers": answers}
    if len(sample_ids) == len(batch):
        result["sample_ids"] = sample_ids
    return result


def reward_MATH(batch, responses, num_generations, device, return_components=False):
    """Compute reward for MATH responses (+1 correct, -1 incorrect)."""
    answers = batch['answers'] * num_generations
    ext_ans = [extract_answer(ans) for ans in answers]
    ext_res = [parse_ground_truth(res)[1] for res in responses]
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(ext_ans, ext_res)):
        if math_equal(ans, res, timeout=True):
            rewards[i] = 1.0
        else:
            rewards[i] = -1.0
    if return_components:
        return rewards, {
            "accuracy_reward": rewards,
            "correct": (rewards > 0).float(),
        }
    return rewards


def load_math_dataset_and_reward(
    local_path,
    batch_size=1,
    split='train',
    num_workers=4,
    seed=112,
    start_position=0,
):
    """Load MATH dataset and return dataloader with reward function."""
    dataset_path = _training_data_path(local_path)
    if dataset_path.is_file():
        ds = _load_local_parquet(dataset_path.resolve(), split)
    elif dataset_path.is_dir():
        local_directory = dataset_path / "data"
        if not local_directory.is_dir():
            local_directory = dataset_path
        ds = _load_local_parquet(local_directory.resolve(), split)
    else:
        ds = load_dataset(local_path, split=split)
    ds = ds.add_column('__sample_id__', list(range(len(ds))))
    ds = ds.filter(lambda x: len(x.get('problem', '')) > 0 and len(x.get('problem', '')) < 1500)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)

    sampler = InfiniteSampler(ds, start_position=start_position)

    dataloader = DataLoader(
        ds,
        collate_fn=collate_fn_math,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )

    return dataloader, reward_MATH
