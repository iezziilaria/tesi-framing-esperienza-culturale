#!/usr/bin/env python3
"""
Baseline: TF-IDF con Logistic Regression e SVM lineare.
Serve da riferimento per il transformer: usa solo le parole e il loro peso.

    python scripts/11_baseline_tfidf.py            cross-validazione a 5 fold sullo sviluppo (641 gold), il test non viene letto
    python scripts/11_baseline_tfidf.py --finale   modello finale: allena su sviluppo + supplementare e valuta sul test

In ogni fold il training e' il gold degli altri fold piu' pilot ed extra (senza i commenti dei
thread della validation); la validation e' solo gold. Il modello finale e' quello indicato in
MODELLO_FINALE, scelto con la macro-F1 media della cross-validazione. Gli extra includono
r/movies (il confronto con e senza e' in docs/esperimenti_preliminari/).

Il test (159 commenti) era gia' stato usato negli esperimenti preliminari: il file dei
risultati lo dichiara.

Esce in modelli/configurazione_iniziale/baseline/ (baseline_cv.json, oppure metriche, figura e previsioni del test).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parent.parent
DATI = BASE / "modelli"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
SEED = 42
N_FOLD = 5
N_BOOTSTRAP = 5000

# modello per la valutazione finale: quello con la macro-F1 media piu' alta nella cross-validazione
MODELLO_FINALE = "logistic_regression"

INCHIOSTRO, INCHIOSTRO_SECONDARIO = "#0b0b0b", "#52514e"

COSTRUTTORI = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED),
    "svm_lineare": lambda: LinearSVC(
        class_weight="balanced", random_state=SEED, dual=True),
}


# lettura dei dati (scritti da 09_split_test_sviluppo_fold.py)

def leggi(nome):
    percorso = DATI / f"{nome}.csv"
    if not percorso.exists():
        raise SystemExit(f"non trovo {percorso}, lancia prima 09_split_test_sviluppo_fold.py")
    return pd.read_csv(percorso).fillna({"sottofonte": ""})


def vettorizzatore():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000,
                            stop_words="english", sublinear_tf=True)


# training e validation di un fold (uguale in 11 e in 12)

def training_del_fold(dev, supp, fold):
    """validation = gold del fold; training = gold degli altri fold + supplementare,
    tolti i commenti che stanno in un thread della validation."""
    va = dev[dev["fold"] == fold]
    gold_tr = dev[dev["fold"] != fold]
    supp_ok = supp[~supp["thread_id"].isin(set(va["thread_id"]))]
    tr = pd.concat([gold_tr, supp_ok], ignore_index=True)
    if set(tr["thread_id"]) & set(va["thread_id"]):
        raise SystemExit(f"fold {fold}: thread condivisi tra training e validation")
    return tr, va


# cross-validazione sui 5 fold del gold

def cross_validazione(dev, supp, costruttore):
    punteggi = []
    for k in range(1, N_FOLD + 1):
        tr, va = training_del_fold(dev, supp, k)
        vec = vettorizzatore()
        X_tr = vec.fit_transform(tr["testo"])
        X_va = vec.transform(va["testo"])
        modello = costruttore()
        modello.fit(X_tr, tr["etichetta"])
        pred = modello.predict(X_va)
        punteggi.append(f1_score(va["etichetta"], pred, labels=CATEGORIE, average="macro", zero_division=0))
    return punteggi


def esegui_cv():
    dev, supp = leggi("dev"), leggi("supplementare")
    risultati = {}
    for nome, costruttore in COSTRUTTORI.items():
        p = cross_validazione(dev, supp, costruttore)
        risultati[nome] = {"media": float(np.mean(p)), "std": float(np.std(p)),
                           "per_fold": [round(x, 3) for x in p]}
    (DATI / "configurazione_iniziale" / "baseline").mkdir(parents=True, exist_ok=True)
    percorso = DATI / "configurazione_iniziale" / "baseline" / "baseline_cv.json"
    if percorso.exists():
        raise SystemExit(f"{percorso} esiste gia': non lo sovrascrivo")
    percorso.write_text(json.dumps(risultati, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- baseline, cross-validazione a 5 fold sul gold (macro-F1) ---")
    for nome, r in risultati.items():
        print(f"{nome:<20} media {r['media']:.3f} (+/- {r['std']:.3f})   per fold {r['per_fold']}")
    print(f"\nscritto in {percorso}")


# modello finale: valutazione sul test (una volta sola) + bootstrap ci

def bootstrap_macro_f1(veri, predetti, n=N_BOOTSTRAP, seed=SEED):
    """IC 95% sulla macro-F1 del test, ricampionando le coppie (vero, predetto)
    con reinserimento."""
    rng = np.random.default_rng(seed)
    veri, predetti = np.array(veri), np.array(predetti)
    valori = []
    for _ in range(n):
        idx = rng.integers(0, len(veri), len(veri))
        valori.append(f1_score(veri[idx], predetti[idx], labels=CATEGORIE, average="macro", zero_division=0))
    return float(np.percentile(valori, 2.5)), float(np.percentile(valori, 97.5))


def figura_matrice_confusione(matrice, percorso):
    scala = LinearSegmentedColormap.from_list("blu", ["#eef4fc", "#1d5aa8"])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.pcolormesh(matrice, cmap=scala, edgecolors="white", linewidth=3, vmin=0, vmax=matrice.max())
    ax.set_xlim(0, 3)
    ax.set_ylim(3, 0)
    for i in range(3):
        for j in range(3):
            v = matrice[i, j]
            colore = "white" if v > matrice.max() * 0.5 else INCHIOSTRO
            ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center", fontsize=15,
                    fontweight="bold" if i == j else "normal", color=colore)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(CATEGORIE)
    ax.set_yticklabels(CATEGORIE)
    ax.set_xlabel("Predetta", color=INCHIOSTRO, fontsize=11, labelpad=8)
    ax.set_ylabel("Vera", color=INCHIOSTRO, fontsize=11, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0, colors=INCHIOSTRO_SECONDARIO, labelsize=10)
    for spina in ax.spines.values():
        spina.set_visible(False)
    fig.tight_layout()
    fig.savefig(percorso, dpi=300, facecolor="white")
    plt.close(fig)


def esegui_finale():
    cartella = DATI / "configurazione_iniziale" / "baseline"
    percorso = cartella / "baseline_valutazione.json"
    if percorso.exists():
        raise SystemExit(f"{percorso} esiste gia': non lo sovrascrivo")
    dev, supp, test = leggi("dev"), leggi("supplementare"), leggi("test")
    training = pd.concat([dev.drop(columns=["fold"]), supp], ignore_index=True)
    if set(training["thread_id"]) & set(test["thread_id"]):
        raise SystemExit("il training contiene thread del test: mi fermo")

    vec = vettorizzatore()
    X_tr = vec.fit_transform(training["testo"])
    X_test = vec.transform(test["testo"])
    modello = COSTRUTTORI[MODELLO_FINALE]()
    modello.fit(X_tr, training["etichetta"])
    pred_test = modello.predict(X_test)

    report = classification_report(test["etichetta"], pred_test, labels=CATEGORIE,
                                    output_dict=True, zero_division=0)
    ic_basso, ic_alto = bootstrap_macro_f1(test["etichetta"].tolist(), list(pred_test))
    matrice = confusion_matrix(test["etichetta"], pred_test, labels=CATEGORIE)

    cartella.mkdir(parents=True, exist_ok=True)
    figura_matrice_confusione(matrice, cartella / "fig_baseline_matrice_confusione.png")
    pd.DataFrame({"id": list(test["id"]), "vero": list(test["etichetta"]), "predetto": list(pred_test)}).to_csv(
        cartella / "baseline_predizioni_test.csv", index=False)
    percorso.write_text(json.dumps({
        "nota_test": "Il test (159 commenti) era gia' stato usato negli esperimenti preliminari.",
        "modello": MODELLO_FINALE, "n_training": len(training),
        "report_test": report, "macro_f1_test": report["macro avg"]["f1-score"],
        "macro_f1_test_ic95_bootstrap": [ic_basso, ic_alto],
        "matrice_confusione_test": {"categorie": CATEGORIE, "valori": matrice.tolist()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"""
--- baseline finale ({MODELLO_FINALE}), training {len(training)} commenti ---
valutazione sul test ({len(test)} commenti), UNA volta sola
{'':14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}""" +
          "".join(f"\n{c:<14}{report[c]['precision']:>10.2f}{report[c]['recall']:>10.2f}"
                  f"{report[c]['f1-score']:>10.2f}{report[c]['support']:>10.0f}" for c in CATEGORIE) +
          f"""

macro-F1 test   {report['macro avg']['f1-score']:.3f}   (IC 95% bootstrap {ic_basso:.3f}-{ic_alto:.3f})
accuracy test   {report['accuracy']:.3f}

scritto in {percorso}
""")


if __name__ == "__main__":
    if "--finale" in sys.argv:
        esegui_finale()
    else:
        esegui_cv()
