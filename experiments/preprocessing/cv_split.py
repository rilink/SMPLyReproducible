import random


def make_folds(subjects: list[str], n_folds: int, subjects_per_fold: int, seed: int) -> list[dict]:
    """Partition `subjects` into `n_folds` disjoint groups of `subjects_per_fold`
    (requires len(subjects) == n_folds * subjects_per_fold). Fold i holds out
    group i for testing and trains on the remaining subjects -- a standard
    leave-N-subjects-out, subject-independent split.
    """
    if len(subjects) != n_folds * subjects_per_fold:
        raise ValueError(
            f"{len(subjects)} subjects does not divide evenly into "
            f"{n_folds} folds of {subjects_per_fold} subjects"
        )

    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)

    folds = []
    for i in range(n_folds):
        test_subjects = sorted(shuffled[i * subjects_per_fold : (i + 1) * subjects_per_fold])
        train_subjects = sorted(s for s in shuffled if s not in test_subjects)
        folds.append({"fold": i, "train_subjects": train_subjects, "test_subjects": test_subjects})
    return folds
