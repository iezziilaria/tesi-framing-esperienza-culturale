#!/usr/bin/env python3
"""
Nuovo protocollo, passo 5: modello finale e valutazione sul test (159 commenti).
E' l'unico script del nuovo protocollo che legge il test, e lo legge una volta.

Legge SOLO modelli_cv_gold/scelte_finali.json (scritto dallo script 23):
condizione (con / senza r/movies), baseline (Logistic Regression o SVM) e numero
di epoche del transformer. Non ha opzioni per cambiarli: le scelte sono quelle
della cross-validation, il test non decide niente.

Cosa allena:
  - training = tutti i 641 gold di sviluppo + pilot ed extra della condizione
    (pilot senza i commenti dei thread del test);
  - baseline: il modello scelto, con lo stesso TF-IDF dello script 11;
  - transformer: per il numero di epoche scelto, ma con lo schedule a 6 epoche
    che avevano i fold (ferma_a), cosi' le condizioni sono quelle viste in CV.

Nota da scrivere in tesi: questo test era gia' stato usato negli esperimenti
preliminari (script 11 e 12, cartella modelli/). Non e' un test visto per la
prima volta. Il file dei risultati lo dichiara.

    python 24_cv_gold_modello_finale.py
    python 24_cv_gold_modello_finale.py --prova    (prova veloce su dati di sviluppo, non legge il test)

Non sovrascrive niente: se i risultati esistono gia' si ferma.
Esce in modelli_cv_gold/finale/ (valutazione_finale.json, predizioni_test.csv,
transformer_checkpoint/).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_gold_comune import (CATEGORIE, RADICE, RMOVIES, SEED, bootstrap_macro_f1,  # noqa: E402
                            leggi_dati, macro_f1, scrivi_json)
from cv_gold_transformer_base import (MODELLO, allena, carica_tokenizer, predici,  # noqa: E402
                                      scegli_device)

COSTRUTTORI = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED),
    "svm_lineare": lambda: LinearSVC(
        class_weight="balanced", random_state=SEED, dual=True),
}


def vettorizzatore():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000,
                            stop_words="english", sublinear_tf=True)


def metriche(veri, predetti):
    report = classification_report(veri, predetti, labels=CATEGORIE, output_dict=True, zero_division=0)
    matrice = confusion_matrix(veri, predetti, labels=CATEGORIE)
    ic = bootstrap_macro_f1(list(veri), list(predetti))
    return {"report": report, "macro_f1": report["macro avg"]["f1-score"],
            "macro_f1_ic95_bootstrap": list(ic),
            "matrice_confusione": {"categorie": CATEGORIE, "valori": matrice.tolist()}}


def stampa(nome, m):
    r = m["report"]
    print(f"\n{nome}")
    print(f"{'':14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for c in CATEGORIE:
        print(f"{c:<14}{r[c]['precision']:>10.2f}{r[c]['recall']:>10.2f}{r[c]['f1-score']:>10.2f}{r[c]['support']:>10.0f}")
    print(f"macro-F1 {m['macro_f1']:.3f}   IC95 bootstrap {m['macro_f1_ic95_bootstrap'][0]:.3f}-{m['macro_f1_ic95_bootstrap'][1]:.3f}"
          f"   accuracy {r['accuracy']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prova", action="store_true")
    args = ap.parse_args()

    dev, supp, test = leggi_dati()
    cartella = RADICE / ("prova" if args.prova else "") / "finale"
    percorso = cartella / "valutazione_finale.json"
    if percorso.exists():
        sys.exit(f"{percorso} esiste gia': non lo sovrascrivo (e il test non si rivaluta per aggiustare le scelte).")

    if args.prova:
        scelte = {"condizione_scelta": "con_rmovies", "epoche_finali": 1,
                  "modello_baseline_finale": "logistic_regression"}
        valutazione = dev[dev["fold"] == 1].sample(24, random_state=SEED)
        training = dev[dev["fold"] != 1].sample(48, random_state=SEED)
        training = pd.concat([training, supp.sample(16, random_state=SEED)], ignore_index=True)
        max_len = 64
        nome_set = "validation del fold 1 (prova, il test non viene letto)"
    else:
        sorgente = RADICE / "scelte_finali.json"
        if not sorgente.exists():
            sys.exit(f"manca {sorgente}: lancia prima 21 e 22 (tutte e due le condizioni) e poi 23")
        scelte = json.loads(sorgente.read_text(encoding="utf-8"))
        s = supp
        if scelte["condizione_scelta"] == "senza_rmovies":
            s = s[s["sottofonte"] != RMOVIES]
        s = s[~s["thread_id"].isin(set(test["thread_id"]))]    # niente thread del test nel training
        training = pd.concat([dev.drop(columns=["fold"]), s], ignore_index=True)
        valutazione = test
        max_len = None
        nome_set = f"test ({len(test)} commenti)"
        if set(training["id"]) & set(test["id"]) or set(training["thread_id"]) & set(test["thread_id"]):
            sys.exit("il training contiene commenti o thread del test: mi fermo")

    print(f"condizione {scelte['condizione_scelta']}   baseline {scelte['modello_baseline_finale']}   "
          f"epoche {scelte['epoche_finali']}   training {len(training)} commenti   valutazione su {nome_set}")

    # baseline
    vec = vettorizzatore()
    X_tr = vec.fit_transform(training["testo"])
    base = COSTRUTTORI[scelte["modello_baseline_finale"]]()
    base.fit(X_tr, training["etichetta"])
    pred_base = list(base.predict(vec.transform(valutazione["testo"])))

    # transformer
    device = scegli_device()
    tokenizer = carica_tokenizer()
    kw = {"max_len": max_len} if max_len else {}
    model, _, pesi = allena(training, None, tokenizer, device, seed=SEED,
                            ferma_a=int(scelte["epoche_finali"]), nome="finale", **kw)
    pred_trans = predici(model, valutazione, tokenizer, device, **kw)

    veri = list(valutazione["etichetta"])
    m_base, m_trans = metriche(veri, pred_base), metriche(veri, pred_trans)

    cartella.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": list(valutazione["id"]), "vero": veri,
                  "baseline": pred_base, "transformer": pred_trans}).to_csv(
        cartella / "predizioni_test.csv", index=False)
    if not args.prova:
        model.save_pretrained(cartella / "transformer_checkpoint")
        tokenizer.save_pretrained(cartella / "transformer_checkpoint")

    scrivi_json(percorso, {
        "nota_test": "Il test (159 commenti) era gia' stato usato negli esperimenti preliminari "
                     "(cartella modelli/): questa non e' una prima valutazione.",
        "prova": args.prova, "scelte_usate": scelte, "modello": MODELLO,
        "n_training": int(len(training)),
        "training_per_fonte": {k: int(v) for k, v in training["fonte"].value_counts().items()},
        "pesi_classe_transformer": {c: round(p, 3) for c, p in zip(CATEGORIE, pesi.tolist())},
        "valutazione_su": nome_set, "n_valutazione": int(len(valutazione)),
        "baseline": m_base, "transformer": m_trans,
    })

    stampa(f"baseline ({scelte['modello_baseline_finale']})", m_base)
    stampa(f"transformer ({MODELLO}, {scelte['epoche_finali']} epoche)", m_trans)
    print(f"\nscritto in {percorso}")


if __name__ == "__main__":
    main()
