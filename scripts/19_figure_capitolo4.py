#!/usr/bin/env python3
"""
Figure del capitolo 4, ricostruite dai file dei risultati (nessun addestramento).

Scrive in docs/figure/:
  - fig_matrici_confronto.(pdf|png): matrici di confusione sul test di baseline e RoBERTa e loro differenza
    (da modelli/configurazione_iniziale/baseline/baseline_predizioni_test.csv e modelli/ricerca_ampliata/3_modelli_finali/roberta/)
  - fig_cv_per_fold.(pdf|png): macro-F1 per fold e media di Logistic Regression e RoBERTa
    (da modelli/configurazione_iniziale/baseline/baseline_cv.json e modelli/ricerca_ampliata/2_transformer_cv/selected_configs.json, media sui tre semi)

Le altre figure della tesi (schemi del capitolo 2 e 3) sono disegni e non sono generate da codice.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix

BASE = Path(__file__).resolve().parent.parent
MODELLI = BASE / "modelli"
RICERCA = MODELLI / "ricerca_ampliata"
CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
INK = "#1f2933"


def matrice(percorso, vero, predetto):
    test = pd.read_csv(MODELLI / "test.csv")[["id", "etichetta"]]
    d = pd.read_csv(percorso).rename(columns={vero: "vero", predetto: "predetto"}).merge(test, on="id")
    assert len(d) == len(test) and (d["vero"] == d["etichetta"]).all()
    return confusion_matrix(d["vero"], d["predetto"], labels=CATEGORIE)


def matrici_test():
    base = matrice(MODELLI / "configurazione_iniziale" / "baseline" / "baseline_predizioni_test.csv", "vero", "predetto")
    rob = matrice(RICERCA / "3_modelli_finali" / "roberta" / "test_predictions.csv", "true", "predicted")
    return base, rob


def figura_matrici(uscita):
    base, rob = matrici_test()
    diff = rob - base
    n = base.sum(axis=1)
    blu = LinearSegmentedColormap.from_list("blu", ["#eef4fc", "#1d5aa8"])
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.1), gridspec_kw={"wspace": 0.55})
    for ax, M, titolo in [(axs[0], base, "Baseline (TF-IDF + Logistic Regression)"), (axs[1], rob, "RoBERTa")]:
        q = M / M.sum(axis=1, keepdims=True)
        ax.imshow(q, vmin=0, vmax=1, cmap=blu)
        for i in range(3):
            for j in range(3):
                col = "white" if q[i, j] > 0.5 else INK
                ax.text(j, i - .08, str(M[i, j]), ha="center", va="center", fontsize=13,
                        fontweight="bold" if i == j else "normal", color=col)
                ax.text(j, i + .27, f"{q[i, j] * 100:.0f}%", ha="center", va="center", fontsize=8.5, color=col)
        ax.set_title(titolo, fontsize=10.5, color=INK, pad=8)
    ax = axs[2]
    for i in range(3):
        for j in range(3):
            d = diff[i, j]
            buono = (d > 0) if i == j else (d < 0)
            fc, tc = ("#F1F3F5", INK) if d == 0 else (("#2F6FB0", "white") if buono else ("#C77A1B", "white"))
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc=fc, ec="white", lw=2))
            ax.text(j, i, "0" if d == 0 else f"{d:+d}".replace("-", "−"), ha="center", va="center",
                    fontsize=15, fontweight="bold" if i == j else "normal", color=tc)
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(2.5, -.5)
    ax.set_aspect("equal")
    ax.set_title("Differenza: RoBERTa − baseline", fontsize=10.5, color=INK, pad=8)
    for k, ax in enumerate(axs):
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CATEGORIE, fontsize=9, rotation=25, ha="right")
        ax.set_yticklabels([f"{c}\n(n = {n[i]})" if k < 2 else c for i, c in enumerate(CATEGORIE)], fontsize=9)
        ax.set_xlabel("Classe predetta", fontsize=9.5, color=INK)
        if k == 0:
            ax.set_ylabel("Classe vera (gold)", fontsize=9.5, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    for estensione, kw in [("png", {"dpi": 300}), ("pdf", {})]:
        fig.savefig(uscita / f"fig_matrici_confronto.{estensione}", bbox_inches="tight", **kw)
    plt.close(fig)


def punteggi_per_fold():
    """Macro-F1 di ogni fold per Logistic Regression e RoBERTa (media sui tre semi)."""
    b = json.loads((MODELLI / "configurazione_iniziale" / "baseline" / "baseline_cv.json").read_text(encoding="utf-8"))["logistic_regression"]["per_fold"]
    sel = json.loads((RICERCA / "2_transformer_cv" / "selected_configs.json").read_text(encoding="utf-8"))["selected"]
    return b, sel["roberta-base"]["fold_means"]


def figura_cv(uscita):
    b, fr = punteggi_per_fold()
    serie = [("Logistic Regression (baseline)", b, "#9AA5B1"), ("RoBERTa", fr, "#1d5aa8")]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    w, x = 0.34, np.arange(6)
    for k, (nome, v, col) in enumerate(serie):
        vals = list(v) + [np.mean(v)]
        ax.bar(x + (k - 0.5) * w, vals, w, color=col, label=nome, edgecolor="white", linewidth=1)
        for xi, vi in zip(x + (k - 0.5) * w, vals):
            ax.text(xi, vi + .008, f"{vi:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axvline(4.5, color="#C9D1D8", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Media"], fontsize=9.5)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Macro-F1 sulla validation", fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.13))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#E4E8EB", lw=.8)
    ax.set_axisbelow(True)
    for estensione, kw in [("png", {"dpi": 300}), ("pdf", {})]:
        fig.savefig(uscita / f"fig_cv_per_fold.{estensione}", bbox_inches="tight", **kw)
    plt.close(fig)


if __name__ == "__main__":
    uscita = BASE / "docs" / "figure"
    uscita.mkdir(parents=True, exist_ok=True)
    figura_matrici(uscita)
    figura_cv(uscita)
    print(f"figure scritte in {uscita}")
