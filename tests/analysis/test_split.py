
import pandas as pd

from analysis.split import multilabel_stratified_kfold, row_multilabels, split_train_validation


def test_row_multilabels_uses_safe_label_for_non_vulnerable_samples():
    assert row_multilabels([]) == {"SAFE"}
    assert row_multilabels(["CWE-079", "89"]) == {"CWE-79", "CWE-89"}


def test_multilabel_stratified_kfold_assigns_all_rows_once():
    df = pd.DataFrame([
        {"sample_id": "a", "cwes": ["CWE-79"]},
        {"sample_id": "b", "cwes": ["CWE-79", "CWE-89"]},
        {"sample_id": "c", "cwes": ["CWE-89"]},
        {"sample_id": "d", "cwes": []},
        {"sample_id": "e", "cwes": []},
        {"sample_id": "f", "cwes": ["CWE-22"]},
    ])

    folds = multilabel_stratified_kfold(df, n_splits=3, random_state=7)

    assert sorted(folds["row_index"].tolist()) == list(range(len(df)))
    assert set(folds["fold"]) == {0, 1, 2}
    assert folds["sample_id"].tolist() == df["sample_id"].tolist()


def test_split_train_validation_uses_fold_assignments():
    df = pd.DataFrame([
        {"sample_id": "a", "cwes": ["CWE-79"]},
        {"sample_id": "b", "cwes": []},
        {"sample_id": "c", "cwes": ["CWE-89"]},
    ])
    folds = pd.DataFrame({"row_index": [0, 1, 2], "fold": [0, 1, 0]})

    train, validation = split_train_validation(df, folds, validation_fold=1)

    assert validation["sample_id"].tolist() == ["b"]
    assert train["sample_id"].tolist() == ["a", "c"]
