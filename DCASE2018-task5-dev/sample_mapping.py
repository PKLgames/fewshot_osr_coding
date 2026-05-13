import pandas as pd
import numpy as np
import os


def split_train_calib_by_session(input_csv, vocab_csv, train_output_csv, calib_output_csv,
                                 num_known_classes=6, num_folds=5, calib_fold_idx=0,
                                 train_samples_per_label=None, calib_known_samples_per_label=None,
                                 calib_unknown_samples_per_label=None, random_state=42):
    """
    Split DCASE2018 fold training data into train and calibration sets by session.

    Splits sessions (not individual segments) into num_folds groups to prevent
    data leakage: all segments from the same session stay together.

    Parameters
    ----------
    input_csv : str — path to foldN_train.txt (filename<TAB>label<TAB>session)
    vocab_csv : str — path to vocabulary.csv
    train_output_csv : str — output path for train CSV
    calib_output_csv : str — output path for calibration CSV
    num_known_classes : int — first N classes in vocab are "known"
    num_folds : int — number of session-level folds
    calib_fold_idx : int — which fold index to use as calibration
    train_samples_per_label : int or None — cap per-label train samples
    calib_known_samples_per_label : int or None — known-class samples in calib
    calib_unknown_samples_per_label : int or None — unknown-class samples in calib
    random_state : int
    """
    # Read vocabulary
    vocab_df = pd.read_csv(vocab_csv, header=None)
    labels_list = vocab_df.iloc[:, 1].tolist()
    all_labels = set(labels_list)
    known_labels = set(labels_list[:num_known_classes])
    unknown_labels = all_labels - known_labels

    print(f"All classes ({len(all_labels)}): {sorted(all_labels)}")
    print(f"Known classes ({len(known_labels)}): {sorted(known_labels)}")
    print(f"Unknown classes ({len(unknown_labels)}): {sorted(unknown_labels)}")

    # Read main data
    df = pd.read_csv(input_csv, delimiter='\t', header=None,
                     names=['filename', 'scene_label', 'session_id'])
    print(f"\nRaw data: {len(df)} records, {df['session_id'].nunique()} sessions")

    # Split sessions per class to guarantee every class appears in calib
    rng = np.random.RandomState(random_state)
    train_sessions = set()
    calib_sessions = set()

    for label in all_labels:
        label_sessions = sorted(df[df['scene_label'] == label]['session_id'].unique())
        rng.shuffle(label_sessions)
        n_sessions = len(label_sessions)

        if n_sessions < num_folds:
            # Edge case: fewer sessions than folds
            # Put at least 1 session in calib if possible, rest in train
            if n_sessions >= 2:
                calib_pick = label_sessions[:1]
                train_pick = label_sessions[1:]
            elif n_sessions == 1:
                calib_pick = label_sessions[:1]
                train_pick = []
            else:
                calib_pick, train_pick = [], []
        else:
            fold_size = n_sessions // num_folds
            calib_start = calib_fold_idx * fold_size
            calib_end = min((calib_fold_idx + 1) * fold_size, n_sessions)
            calib_pick = label_sessions[calib_start:calib_end]
            train_pick = [s for s in label_sessions if s not in set(calib_pick)]

        calib_sessions.update(calib_pick)
        if label in known_labels:
            train_sessions.update(train_pick)
        # unknown classes: sessions not used in calib are discarded (not added to train)

    print(f"\nSession split: train={len(train_sessions)}, calib={len(calib_sessions)}")

    # Build train: known classes from train sessions
    train_df = df[df['session_id'].isin(train_sessions) & df['scene_label'].isin(known_labels)].copy()
    # Build calib: all classes from calib sessions
    calib_df = df[df['session_id'].isin(calib_sessions)].copy()

    print(f"\nBefore sampling:")
    print(f"  Train: {len(train_df)} records "
          f"({train_df['session_id'].nunique()} sessions, "
          f"{train_df['scene_label'].nunique()} classes)")
    print(f"  Calib: {len(calib_df)} records "
          f"({calib_df['session_id'].nunique()} sessions, "
          f"{calib_df['scene_label'].nunique()} classes)")

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    calib_df = calib_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Optional: cap train samples per label
    if train_samples_per_label is not None:
        print(f"\nCapping train samples to {train_samples_per_label} per label...")
        sampled = []
        for label in known_labels:
            group = train_df[train_df['scene_label'] == label]
            n = min(train_samples_per_label, len(group))
            sampled.append(group.sample(n=n, random_state=random_state, replace=False))
        train_df = pd.concat(sampled, ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"  Train after capping: {len(train_df)} records")

    # Optional: cap calib samples (few-shot for unknown)
    if calib_known_samples_per_label is not None or calib_unknown_samples_per_label is not None:
        n_known = calib_known_samples_per_label or calib_known_samples_per_label
        n_unknown = calib_unknown_samples_per_label or calib_unknown_samples_per_label
        print(f"\nCapping calib: known={n_known}/class, unknown={n_unknown}/class")
        sampled = []
        for label in all_labels:
            group = calib_df[calib_df['scene_label'] == label]
            n = n_known if label in known_labels else n_unknown
            n = min(n, len(group))
            if n > 0:
                sampled.append(group.sample(n=n, random_state=random_state, replace=False))
        calib_df = pd.concat(sampled, ignore_index=True)
        calib_df = calib_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"  Calib after capping: {len(calib_df)} records")

    # Save
    os.makedirs(os.path.dirname(train_output_csv), exist_ok=True)
    train_df.to_csv(train_output_csv, index=False, sep='\t')
    calib_df.to_csv(calib_output_csv, index=False, sep='\t')

    print(f"\nDone!")
    print(f"  Train: {len(train_df)} records → {train_output_csv}")
    print(f"  Calib: {len(calib_df)} records → {calib_output_csv}")
    print(f"\nTrain label distribution:")
    print(train_df['scene_label'].value_counts().sort_index())
    print(f"\nCalib label distribution:")
    print(calib_df['scene_label'].value_counts().sort_index())

    return train_df, calib_df


def sample_test_set(evaluate_csv, vocab_csv, output_csv,
                    samples_per_label=None, random_state=42):
    """
    Process the evaluate file into a test set with all classes.

    Parameters
    ----------
    evaluate_csv : str — path to foldN_evaluate.txt
    vocab_csv : str — path to vocabulary.csv
    output_csv : str — output path
    samples_per_label : int or None — optional per-class cap
    random_state : int
    """
    vocab_df = pd.read_csv(vocab_csv, header=None)
    all_labels = set(vocab_df.iloc[:, 1].tolist())

    df = pd.read_csv(evaluate_csv, delimiter='\t', header=None,
                     names=['filename', 'scene_label'])
    print(f"Evaluate data: {len(df)} records, {df['scene_label'].nunique()} classes")

    if samples_per_label is not None:
        print(f"Capping test to {samples_per_label} per label...")
        sampled = []
        for label in all_labels:
            group = df[df['scene_label'] == label]
            n = min(samples_per_label, len(group))
            if n > 0:
                sampled.append(group.sample(n=n, random_state=random_state, replace=False))
        df = pd.concat(sampled, ignore_index=True)
        print(f"  After capping: {len(df)} records")

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, sep='\t')
    print(f"Test set: {len(df)} records → {output_csv}")
    print(df['scene_label'].value_counts().sort_index())
    return df


if __name__ == "__main__":
    BASE = "/coding/DCASE2018-task5-dev"
    vocab_file = f"{BASE}/sampled_setup/vocabulary.csv"
    num_known = 5  # first 5 classes in vocab are known

    for fold_id in range(1, 5):
        print(f"\n{'=' * 70}")
        print(f"Processing Fold {fold_id}")
        print(f"{'=' * 70}")

        input_train = f"{BASE}/evaluation_setup/fold{fold_id}_train.txt"
        input_eval = f"{BASE}/evaluation_setup/fold{fold_id}_evaluate.txt"

        out_train = f"{BASE}/sampled_setup/fold{fold_id}_train.csv"
        out_calib = f"{BASE}/sampled_setup/fold{fold_id}_calib.csv"
        out_test = f"{BASE}/sampled_setup/fold{fold_id}_evaluate.csv"

        # Step 1: Split train into train (known only) + calib (all classes)
        split_train_calib_by_session(
            input_csv=input_train,
            vocab_csv=vocab_file,
            train_output_csv=out_train,
            calib_output_csv=out_calib,
            num_known_classes=num_known,
            num_folds=5,
            calib_fold_idx=0,
            train_samples_per_label=6000,
            calib_known_samples_per_label=30,
            calib_unknown_samples_per_label=3,
            random_state=42,
        )

        # Step 2: Process test set (all classes)
        sample_test_set(
            evaluate_csv=input_eval,
            vocab_csv=vocab_file,
            output_csv=out_test,
            samples_per_label=500,
            random_state=456,
        )
