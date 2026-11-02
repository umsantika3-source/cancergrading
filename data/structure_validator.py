"""
structure_validator.py — Validates that the dataset folder at data_root
matches the structure type selected in config.DATA_STRUCTURE.

If validation fails, returns (False, <detailed message>) so the caller
can stop the flow and notify the user.
"""

import os
from typing import Tuple


def validate_structure(
    data_root: str,
    structure_type: str,
    k_folds: int = 5,
) -> Tuple[bool, str]:
    """
    Validate that data_root matches the expected folder structure.

    Args:
        data_root: Root directory of the dataset.
        structure_type: "old" or "kfold5".
        k_folds: Number of folds expected when structure_type == "kfold5".

    Returns:
        (True, "OK") if the structure matches.
        (False, <message>) if it does not, with details on what was expected
        vs. what was found.
    """
    if not os.path.isdir(data_root):
        return False, (
            f"DATA_ROOT does not exist or is not a directory: {data_root}\n"
            f"Please update config.INPUT_DIR to point to your dataset root."
        )

    try:
        top_entries = sorted(os.listdir(data_root))
    except PermissionError:
        top_entries = []

    if structure_type == "old":
        return _validate_old_structure(data_root, top_entries)
    elif structure_type == "kfold5":
        return _validate_kfold5_structure(data_root, top_entries, k_folds)
    else:
        return False, (
            f"Unknown DATA_STRUCTURE value: '{structure_type}'.\n"
            f"Valid values are: 'old', 'kfold5'.\n"
            f"Please update config.DATA_STRUCTURE."
        )


def _validate_old_structure(data_root: str, top_entries: list) -> Tuple[bool, str]:
    """Old structure: data_root/train/, data_root/val/, data_root/test/
    with grade subfolders inside each split."""
    required_splits = ["train", "val", "test"]
    missing = [s for s in required_splits if not os.path.isdir(os.path.join(data_root, s))]

    if missing:
        expected = "\n".join(
            f"  {split}/Grade1/\n  {split}/Grade2/\n  {split}/Grade3/"
            for split in required_splits
        )
        found = "\n".join(f"  {e}/" for e in top_entries) if top_entries else "  (empty or inaccessible)"
        return False, (
            f"Structure validation failed for DATA_STRUCTURE='old'.\n"
            f"Root: {data_root}\n\n"
            f"Missing required split folder(s): {missing}\n\n"
            f"Expected structure:\n{expected}\n\n"
            f"Found at root:\n{found}\n\n"
            f"Please either:\n"
            f"  1. Point config.INPUT_DIR to a folder with train/val/test layout, or\n"
            f"  2. Set config.DATA_STRUCTURE = 'kfold5' if you are using the k-fold dataset."
        )

    train_dir = os.path.join(data_root, "train")
    grade_entries = [
        e for e in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, e))
    ]
    grade_prefixes = ("Grade1", "Grade2", "Grade3", "Grade_1", "Grade_2", "Grade_3")
    has_grades = any(g.startswith(grade_prefixes) for g in grade_entries)
    if not has_grades:
        return False, (
            f"Structure validation failed for DATA_STRUCTURE='old'.\n"
            f"Root: {data_root}\n\n"
            f"Split folders exist (train/, val/, test/) but no grade subfolders "
            f"were found under train/.\n"
            f"Expected: train/Grade1/, train/Grade2/, train/Grade3/\n"
            f"Found under train/: {grade_entries if grade_entries else '(none)'}\n\n"
            f"Please ensure images are organized into grade subfolders."
        )

    return True, "OK"


def _validate_kfold5_structure(
    data_root: str, top_entries: list, k_folds: int
) -> Tuple[bool, str]:
    """Kfold5 structure: data_root/fold_1/ … data_root/fold_K/,
    each with train/ and val/, each containing Grade_1/, Grade_2/, Grade_3/."""
    fold_entries = [e for e in top_entries if e.startswith("fold_")]

    if not fold_entries:
        expected = "\n".join(
            f"  fold_{i}/train/Grade_{g}/\n  fold_{i}/val/Grade_{g}/"
            for i in range(1, k_folds + 1) for g in (1, 2, 3)
        )
        found = "\n".join(f"  {e}/" for e in top_entries) if top_entries else "  (empty or inaccessible)"
        return False, (
            f"Structure validation failed for DATA_STRUCTURE='kfold5'.\n"
            f"Root: {data_root}\n\n"
            f"No 'fold_N' directories found.\n\n"
            f"Expected structure (k_folds={k_folds}):\n{expected}\n\n"
            f"Found at root:\n{found}\n\n"
            f"Please either:\n"
            f"  1. Point config.INPUT_DIR to the kfold_dataset root containing fold_1/…fold_{k_folds}/, or\n"
            f"  2. Set config.DATA_STRUCTURE = 'old' if you are using the train/val/test layout."
        )

    expected_folds = [f"fold_{i}" for i in range(1, k_folds + 1)]
    missing_folds = [f for f in expected_folds if f not in fold_entries]
    if missing_folds:
        return False, (
            f"Structure validation failed for DATA_STRUCTURE='kfold5'.\n"
            f"Root: {data_root}\n\n"
            f"Missing fold(s): {missing_folds}\n"
            f"Expected folds: {expected_folds}\n"
            f"Found folds: {sorted(fold_entries)}\n\n"
            f"Please ensure all {k_folds} folds are present, or reduce config.K_FOLDS."
        )

    grade_prefixes = ("Grade_1", "Grade_2", "Grade_3")
    problems = []
    for fold_name in expected_folds:
        fold_dir = os.path.join(data_root, fold_name)
        for split in ("train", "val"):
            split_dir = os.path.join(fold_dir, split)
            if not os.path.isdir(split_dir):
                problems.append(f"  {fold_name}/{split}/  (missing)")
                continue
            split_entries = [
                e for e in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, e))
            ]
            missing_grades = [
                g for g in grade_prefixes
                if not any(e.startswith(g) for e in split_entries)
            ]
            if missing_grades:
                problems.append(
                    f"  {fold_name}/{split}/ missing grade subfolders: {missing_grades}\n"
                    f"    Found: {split_entries if split_entries else '(none)'}"
                )

    if problems:
        prob_text = "\n".join(problems)
        expected = (
            f"  fold_N/train/Grade_1/, Grade_2/, Grade_3/\n"
            f"  fold_N/val/Grade_1/,   Grade_2/,   Grade_3/"
        )
        return False, (
            f"Structure validation failed for DATA_STRUCTURE='kfold5'.\n"
            f"Root: {data_root}\n\n"
            f"Issues found inside fold directories:\n{prob_text}\n\n"
            f"Expected per fold:\n{expected}\n\n"
            f"Please fix the folder layout or adjust config.K_FOLDS / config.DATA_STRUCTURE."
        )

    return True, "OK"