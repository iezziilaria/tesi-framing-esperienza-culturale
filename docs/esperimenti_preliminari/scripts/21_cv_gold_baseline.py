#!/usr/bin/env python3
"""
Nuovo protocollo, passo 2: cross-validation della baseline (TF-IDF + Logistic
Regression e SVM lineare) sui 5 fold gold preparati dallo script 20.

Si lancia una volta per condizione:
    python 21_cv_gold_baseline.py --condizione con_rmovies
    python 21_cv_gold_baseline.py --condizione senza_rmovies

In ogni fold: training = gold degli altri fold + pilot ed extra (senza i
commenti dei thread della validation); validation = solo gold del fold.
Il TF-IDF si rifa' da zero sul training del fold. Vettorizzatore e modelli
sono gli stessi dello script 11 (copiati, non cambiati).

Il test non viene letto: serve solo per sapere quali thread tenere fuori.

Esce in modelli_cv_gold/cv/<condizione>/baseline_cv.json e
predizioni_baseline.csv (le previsioni di ogni validation, per l'analisi errori).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_gold_comune import (CATEGORIE, CONDIZIONI, N_FOLD, SEED, cartella_cv, leggi_dati,  # noqa: E402
                            macro_f1, scrivi_json, training_del_fold)

# copiati dallo script 11: non cambiarli, altrimenti i confronti non valgono piu'
COSTRUTTORI = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED),
    "svm_lineare": lambda: LinearSVC(
        class_weight="balanced", random_state=SEED, dual=True),
}


def vettorizzatore():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000,
                            stop_words="english", sublinear_tf=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condizione", required=True, choices=CONDIZIONI)
    args = ap.parse_args()

    cartella = cartella_cv(args.condizione)
    percorso = cartella / "baseline_cv.json"
    if percorso.exists():
        sys.exit(f"{percorso} esiste gia': non lo sovrascrivo.")

    dev, supp, test = leggi_dati()

    punteggi = {nome: [] for nome in COSTRUTTORI}
    predizioni = []          # una riga per commento di validation e per modello
    composizione = []
    for k in range(1, N_FOLD + 1):
        tr, va, info = training_del_fold(dev, supp, test, k, args.condizione)
        composizione.append(info)
        vec = vettorizzatore()
        X_tr = vec.fit_transform(tr["testo"])
        X_va = vec.transform(va["testo"])
        for nome, costruttore in COSTRUTTORI.items():
            modello = costruttore()
            modello.fit(X_tr, tr["etichetta"])
            pred = modello.predict(X_va)
            punteggi[nome].append(macro_f1(va["etichetta"], pred))
            for i, v, p in zip(va["id"], va["etichetta"], pred):
                predizioni.append({"id": i, "fold": k, "modello": nome, "vero": v, "predetto": p})
        print(f"fold {k}/{N_FOLD}: " + "   ".join(f"{n} {punteggi[n][-1]:.3f}" for n in COSTRUTTORI), flush=True)

    pred_df = pd.DataFrame(predizioni)
    risultati = {}
    for nome in COSTRUTTORI:
        p = pred_df[pred_df["modello"] == nome]
        report = classification_report(p["vero"], p["predetto"], labels=CATEGORIE,
                                        output_dict=True, zero_division=0)
        matrice = confusion_matrix(p["vero"], p["predetto"], labels=CATEGORIE)
        risultati[nome] = {
            "macro_f1_per_fold": [round(x, 4) for x in punteggi[nome]],
            "media": float(np.mean(punteggi[nome])),
            "std": float(np.std(punteggi[nome])),
            "std_ddof1": float(np.std(punteggi[nome], ddof=1)),
            "report_validation_aggregata": report,
            "matrice_confusione_aggregata": {"categorie": CATEGORIE, "valori": matrice.tolist()},
        }

    pred_df.to_csv(cartella / "predizioni_baseline.csv", index=False)
    scrivi_json(percorso, {
        "condizione": args.condizione, "seed_modelli": SEED, "n_fold": N_FOLD,
        "metrica_primaria": "macro-F1 (media sui fold)",
        "modelli": risultati, "composizione_per_fold": composizione,
    })

    print(f"\n--- baseline, 5 fold sul gold, condizione {args.condizione} ---")
    for nome, r in risultati.items():
        print(f"{nome:<20} macro-F1 media {r['media']:.3f} (+/- {r['std']:.3f})   per fold {r['macro_f1_per_fold']}")
    print(f"\nscritto in {percorso}")


if __name__ == "__main__":
    main()
