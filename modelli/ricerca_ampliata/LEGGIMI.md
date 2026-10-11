# Ricerca ampliata: dove trovare le cose

Qui ci sono i risultati dei notebook `scripts/14` – `scripts/16` (e, in `1_baseline_C/`, un controllo supplementare sulla baseline con il suo notebook) (eseguiti su Google Colab e Kaggle, con GPU).
La ricerca è stata fatta **solo sullo sviluppo** (641 commenti del gold + materiale supplementare, 5 fold).
Il test (159 commenti) si legge una sola volta, alla fine, nella cartella `3_modelli_finali/`.

I dati usati sono quelli di `modelli/dev.csv`, `supplementare.csv` e `test.csv` (identici byte per byte).
I nomi dei file e le colonne sono in inglese perché vengono dai notebook.

## Le cartelle, in ordine

| Cartella | Notebook | Cosa contiene |
|---|---|---|
| `1_baseline_C/` | `notebook_baseline_C_colab.ipynb` (nella cartella) | Controllo supplementare sulla baseline: $C$ (0,1, 1, 10) per Logistic Regression e SVM sui 5 fold, 30 addestramenti ($C = 1$ risulta il migliore). Non riportato nella tesi |
| `2_transformer_cv/` | `14` | Ricerca per RoBERTa e DistilRoBERTa: 3 velocità di apprendimento × 3 semi × 5 fold, con 6 epoche (90 addestramenti) |
| `3_modelli_finali/` | `15` | I tre modelli scelti, riaddestrati su 1.068 commenti e valutati **sul test** |
| `4_early_stopping_fold1/` | `16` | Prova a parte: RoBERTa con un solo fold di validation, fino a 30 epoche, arresto anticipato |

`thesis_ml_kaggle_inputs.zip` è il pacchetto di input da caricare su Kaggle per i notebook `15` e `16`
(contiene i tre CSV e i risultati di `1_` e `2_`; istruzioni in `docs/ricerca_ampliata/KAGGLE_README.md`).

## Che file aprire per ogni numero della tesi

| Numero | File |
|---|---|
| $C$ della baseline (0,547 / 0,554 / 0,550 per la Logistic Regression) | `1_baseline_C/candidate_summary.csv` |
| Scelta di velocità di apprendimento ed epoca (RoBERTa: 5e-5, epoca 6, 0,679; DistilRoBERTa: 2e-5, epoca 5, 0,638) | `2_transformer_cv/selected_configs.json` |
| Tutte le 36 combinazioni velocità × epoca, con media e deviazioni (tabella in appendice) | `2_transformer_cv/candidate_summary.csv` |
| Ogni singola misura (modello, velocità, seme, fold, epoca) | `2_transformer_cv/run_metrics.csv` (540 righe) |
| Curve per epoca | `2_transformer_cv/*_learning_curves.png` |
| Risultati sul test dei tre modelli (macro-F1, accuratezza, F1 per classe) | `3_modelli_finali/final_comparison.csv` |
| Dettaglio per classe e matrice di confusione di RoBERTa | `3_modelli_finali/roberta/evaluation.json` e `confusion_matrix.png` |
| Predizioni sul test di ciascun modello | `3_modelli_finali/<modello>/test_predictions.csv` |
| Prova con più epoche (migliore all'epoca 8, 0,737) | `4_early_stopping_fold1/epoch_metrics.csv` e `selection.json` |

## Come sono fatte le cartelle dentro

- `runs/<modello>/lr_<velocità>/seed_<seme>/fold_<k>/`: metriche e predizioni di ogni singolo addestramento.
- `manifest.json`, `source_*_manifest.json`, `progress.json`: dati tecnici per tracciare ogni esecuzione (versioni delle librerie, impronte dei file, tempi). I percorsi che compaiono dentro sono quelli di Colab e Kaggle.
- `checkpoint/` (in `3_modelli_finali/roberta/` e `distilroberta/`): configurazione e tokenizer del modello. I **pesi** (`model.safetensors`, circa 500 MB l'uno) non sono inclusi: si copiano a parte e sono ignorati da git.

## Cose da sapere

- Il test non è stato usato per scegliere: velocità di apprendimento, epoca, $C$ e modello si scelgono con la cross-validazione (media su 5 fold e 3 semi, stessa epoca in tutti i fold).
- Il test era già stato consultato prima, e il modello della configurazione iniziale (`modelli/configurazione_iniziale/`) era stato valutato sullo stesso test: i due risultati si confrontano in tesi (§4.1).
- La ricerca è stata eseguita in più sessioni Colab; qui ci sono i risultati dell'esportazione completa (90 addestramenti per i transformer).
- Il seme del modello finale è 42, fissato prima di leggere il test. I semi 42, 123, 2026 nella ricerca servono a misurare la variabilità, non a scegliere.
- Ripetere gli stessi addestramenti con un'altra GPU o altre versioni delle librerie può dare risultati leggermente diversi.

## Quale modello ha prodotto i risultati della tesi

- **Test e corpus**: il RoBERTa di `3_modelli_finali/roberta/` (velocità di apprendimento 5e-5, 6 epoche, seme 42, addestrato su Kaggle sui 1.068 commenti di training). Il file dei pesi è `3_modelli_finali/roberta/checkpoint/model.safetensors` (SHA-256 `ba272a4355074e6d9fb739190214a16db7115aa415956d56d200b10d6792a480`).
- L'applicazione al corpus (`scripts/17_applicazione_corpus.py`, su CPU) usa questo checkpoint; riapplicato ai 159 commenti del test riproduce le predizioni di `test_predictions.csv` (159 su 159).
- Il RoBERTa della configurazione iniziale (velocità 2e-5, 4 epoche) è in `modelli/configurazione_iniziale/roberta/` e non ha prodotto nessuna delle predizioni sul corpus riportate nella tesi (le sue sono in `data/processed/*_iniziale.*`).
