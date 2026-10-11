#!/usr/bin/env python3
"""
Baseline: TF-IDF + Logistic Regression / SVM lineare.

Serve come punto di riferimento prima del modello transformer: se il
transformer non fa meglio di questa baseline, vuol dire che non sta imparando
niente in piu' del semplice lessico. Senza baseline non ci sarebbe modo di
saperlo.

Procedura (aggiornata su richiesta della relatrice, che ha chiesto una k-fold
stratificata invece di un singolo validation set, piu' robusta quando una
classe e' scarsa):
  1. train e validation (script 10) vengono uniti in un unico "pool" di
     training (822 commenti).
  2. per ciascun modello (Logistic Regression, SVM lineare), 5-fold
     StratifiedGroupKFold sul pool: raggruppato per thread (un thread non
     spacca mai un fold) e stratificato per etichetta. In ogni fold il TF-IDF
     si rifa' da zero, fit solo sulla parte di training del fold - mai su
     quella di validation, stessa regola di prima.
  3. si sceglie il modello con la macro-F1 media piu' alta sui 5 fold (non su
     un singolo split, che con lo strumentale cosi' scarso puo' essere
     fortunato o sfortunato per caso).
  4. il modello scelto viene riaddestrato UNA volta su tutto il pool (822) e
     valutato UNA volta sul test (mai visto finora): precision/recall/F1 per
     classe, macro-F1, matrice di confusione, piu' un intervallo di confidenza
     bootstrap al 95% sulla macro-F1 del test (per dare una misura di
     stabilita' della stima senza dover rifare il training piu' volte).

Esce in modelli/baseline_tfidf_metriche.json e
docs/figure/fig_baseline_matrice_confusione.png.
"""

import json
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parent.parent
MODELLI = BASE / "modelli"
FIGURE = BASE / "docs" / "figure"
USCITA_JSON = MODELLI / "baseline_tfidf_metriche.json"

CATEGORIE = ["strumentale", "intrinseco", "ambiguo"]
SEED = 42
FOLD_CV = 5
N_BOOTSTRAP = 5000

INCHIOSTRO, INCHIOSTRO_SECONDARIO = "#0b0b0b", "#52514e"

COSTRUTTORI = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED),
    "svm_lineare": lambda: LinearSVC(
        class_weight="balanced", random_state=SEED, dual=True),
}


# ============================================================
# 1. LETTURA DEGLI SPLIT (scritti da 10_split_train_val_test.py)
# ============================================================

def leggi_split(nome):
    percorso = MODELLI / f"{nome}.csv"
    if not percorso.exists():
        raise SystemExit(f"non trovo {percorso}, lancia prima 10_split_train_val_test.py")
    return pd.read_csv(percorso)


def vettorizzatore():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000,
                            stop_words="english", sublinear_tf=True)


# ============================================================
# 2. CROSS-VALIDAZIONE STRATIFICATA E RAGGRUPPATA PER THREAD
# ============================================================

def cross_validazione(pool, costruttore, n_fold, seed):
    y = pool["etichetta"].to_numpy()
    gruppi = pool["thread_id"].to_numpy()
    skf = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    punteggi = []
    for idx_tr, idx_va in skf.split(np.zeros(len(pool)), y, gruppi):
        tr, va = pool.iloc[idx_tr], pool.iloc[idx_va]
        vec = vettorizzatore()
        X_tr = vec.fit_transform(tr["testo"])
        X_va = vec.transform(va["testo"])
        modello = costruttore()
        modello.fit(X_tr, tr["etichetta"])
        pred = modello.predict(X_va)
        punteggi.append(f1_score(va["etichetta"], pred, labels=CATEGORIE, average="macro", zero_division=0))
    return punteggi


# ============================================================
# 3. VALUTAZIONE FINALE SUL TEST (una volta sola) + bootstrap CI
# ============================================================

def bootstrap_macro_f1(veri, predetti, n=N_BOOTSTRAP, seed=SEED):
    """IC 95% sulla macro-F1 del test, ricampionando le coppie (vero, predetto)
    con reinserimento - da' un'idea di quanto la stima sarebbe stata diversa
    con un test leggermente diverso, senza dover riallenare niente."""
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


def main():
    train, val, test = leggi_split("train"), leggi_split("val"), leggi_split("test")
    pool = pd.concat([train, val], ignore_index=True)

    # 1-2. cross-validazione stratificata e raggruppata, per ciascun modello
    cv_risultati = {}
    for nome, costruttore in COSTRUTTORI.items():
        punteggi = cross_validazione(pool, costruttore, FOLD_CV, SEED)
        cv_risultati[nome] = {"media": float(np.mean(punteggi)), "std": float(np.std(punteggi)),
                              "per_fold": [round(p, 3) for p in punteggi]}
    vincitore = max(cv_risultati, key=lambda n: cv_risultati[n]["media"])

    # 3. modello scelto: riaddestrato su tutto il pool, valutato una volta sul test
    vec = vettorizzatore()
    X_pool = vec.fit_transform(pool["testo"])
    X_test = vec.transform(test["testo"])
    modello = COSTRUTTORI[vincitore]()
    modello.fit(X_pool, pool["etichetta"])
    pred_test = modello.predict(X_test)

    report = classification_report(test["etichetta"], pred_test, labels=CATEGORIE,
                                    output_dict=True, zero_division=0)
    macro_f1_test = report["macro avg"]["f1-score"]
    ic_basso, ic_alto = bootstrap_macro_f1(test["etichetta"].tolist(), list(pred_test))
    matrice = confusion_matrix(test["etichetta"], pred_test, labels=CATEGORIE)

    FIGURE.mkdir(parents=True, exist_ok=True)
    figura_matrice_confusione(matrice, FIGURE / "fig_baseline_matrice_confusione.png")

    metriche = {
        "vocabolario_tfidf_pool": len(vec.vocabulary_),
        "cross_validazione_5fold": cv_risultati,
        "modello_scelto": vincitore,
        "report_test": report,
        "macro_f1_test": macro_f1_test,
        "macro_f1_test_ic95_bootstrap": [ic_basso, ic_alto],
        "matrice_confusione_test": {"categorie": CATEGORIE, "valori": matrice.tolist()},
        "n_pool_train_val": len(pool), "n_test": len(test),
    }
    USCITA_JSON.write_text(json.dumps(metriche, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"""
--- baseline TF-IDF, con 5-fold cross-validation ---
pool di training (train+val uniti)   {len(pool)} commenti

macro-F1 media sui 5 fold (per scegliere il modello):
  logistic regression      {cv_risultati['logistic_regression']['media']:.3f}  (+/- {cv_risultati['logistic_regression']['std']:.3f})   fold: {cv_risultati['logistic_regression']['per_fold']}
  svm lineare               {cv_risultati['svm_lineare']['media']:.3f}  (+/- {cv_risultati['svm_lineare']['std']:.3f})   fold: {cv_risultati['svm_lineare']['per_fold']}
  -> scelto: {vincitore}

--- valutazione sul test ({len(test)} commenti), UNA volta sola, modello riaddestrato su tutto il pool ---
{'':12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}""" +
          "".join(f"\n{c:<12}{report[c]['precision']:>10.2f}{report[c]['recall']:>10.2f}"
                  f"{report[c]['f1-score']:>10.2f}{report[c]['support']:>10.0f}" for c in CATEGORIE) +
          f"""

macro-F1 test               {macro_f1_test:.3f}   (IC95% bootstrap {ic_basso:.3f} - {ic_alto:.3f})
accuracy test                {report['accuracy']:.3f}

figura in {FIGURE / 'fig_baseline_matrice_confusione.png'}
metriche in {USCITA_JSON}
""")


if __name__ == "__main__":
    main()
