# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import zipfile
import numpy as np
import pandas as pd

from GSSL import AGR, EAGR, fFME, rFME, MiMoLaP, DDGL, MFAGL, vGMMGL, vMFAGL
from utils import draw_labels

algorithms = {
    "agr": AGR,
    "eagr": EAGR,
    "ffme": fFME,
    "rfme": rFME,
    "mimolaplg": MiMoLaP,
    "mimolaph": MiMoLaP,
    "ddgl": DDGL,
    "mfagl": MFAGL,
    "vgmmgl": vGMMGL,
    "vmfagl": vMFAGL,
}

names = {
    "agr": "AGR",
    "eagr": "EAGR",
    "ffme": "f-FME",
    "rfme": "r-FME",
    "mimolaplg": "MiMoLaPᴸᴳ",
    "mimolaph": "MiMoLaPᴴ",
    "ddgl": "DDGL",
    "mfagl": "MFA-GL",
    "vgmmgl": "v-GMMᵈ-GL",
    "vmfagl": "v-MFA-GL",
}


def read_data():
    zf = zipfile.ZipFile("./data/letter+recognition.zip")
    letter = pd.read_csv(
        zf.open("letter-recognition.data"),
        header=None,
    )
    labels = letter[0].to_numpy()
    labels_grouped = np.argwhere(labels == np.unique(labels)[:, np.newaxis])
    labels_rel = labels_grouped[np.argsort(labels_grouped[:, 1]), 0]
    data = letter.drop(columns=[letter.columns[0]]).to_numpy()
    data = np.ascontiguousarray(data, dtype=np.float64)
    return data, labels_rel


def run(model_name, X, labels, label_train, C, rng):
    N, D = X.shape

    model = algorithms[model_name](C, D)
    kwargs = (
        dict(mode=model_name[7:]) if model_name in ("mimolaplg", "mimolaph") else {}
    )

    predicted_labels = model.fit_predict(X, label_train, rng=rng, **kwargs)

    acc = np.mean(predicted_labels[label_train == -1] == labels[label_train == -1])
    print(f"Error-Rate of {names[model_name]} on Letter is {(1 - acc) * 100:.2f}%")


if __name__ == "__main__":
    models = [
        "agr",
        "eagr",
        "ffme",
        "rfme",
        "mimolaplg",
        "mimolaph",
        "ddgl",
        "mfagl",
        "vgmmgl",
        "vmfagl",
    ]
    C = 500  # number of components/clusters
    Nl = 100  # number of labeled data points

    seed = 100  # for random number generation

    rng = np.random.default_rng(seed)

    data, labels = read_data()

    label_names = np.unique(labels)
    label_train = draw_labels(labels, label_names, Nl, even_labels=True, rng=rng)

    for model_name in models:
        run(model_name, data, labels, label_train, C, seed)
