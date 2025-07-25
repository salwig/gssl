# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import numpy as np


def draw_labels(labels, label_names, label_size, even_labels=False, rng=None):
    rng = np.random.default_rng(rng)
    N = len(labels)
    train_labels = (-1) * np.ones(N, dtype=np.int64)
    if even_labels:
        number_label_names = len(label_names)

        labels_per_class = np.repeat(
            int(label_size / number_label_names), number_label_names
        )
        random_labels = label_size % number_label_names

        labels_per_class[
            rng.choice(label_names, size=random_labels, replace=False)
        ] += 1
        assert labels_per_class.sum() == label_size

        sorted_ind = {l: np.where(labels == l)[0] for l in label_names}

        for (label, indices), label_per_class in zip(
            sorted_ind.items(), labels_per_class
        ):
            train_labels[rng.choice(indices, size=label_per_class, replace=False)] = (
                label
            )
        assert len(np.nonzero(train_labels + 1)[0]) == label_size
    else:
        indices = rng.choice(np.arange(N), size=label_size, replace=False)
        train_labels[indices] = labels[indices]

    return train_labels
