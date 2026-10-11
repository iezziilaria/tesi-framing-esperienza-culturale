# Framing dell'esperienza culturale: riconoscerlo dai commenti con NLP

Codice e dati della tesi triennale "Riconoscere il framing dell'esperienza culturale attraverso il NLP" (UniBo).
Il compito: dato un commento su un film (r/TrueFilm), dire se presenta la visione come **strumentale**
(mezzo per un risultato), **intrinseca** (valore in sé) o **ambigua / non classificabile**.
Si confrontano una baseline lessicale (TF-IDF con Logistic Regression e SVM) e RoBERTa con fine-tuning.

## Come è organizzata la cartella
```
scripts/                 la pipeline, numerata nell'ordine di esecuzione (script .py e notebook .ipynb)
data/                    dati di partenza e annotazioni (corpus, pilot, gold, materiale aggiuntivo)
modelli/                 split, risultati e previsioni dei modelli (configurazione iniziale e ricerca ampliata)
docs/                    manuale di annotazione, figure della tesi, analisi degli errori, relazione sulla ricerca ampliata, esperimenti preliminari
requirements.txt         versioni delle librerie per gli script locali (00-13 e 17-19)
requirements_gpu.txt     versioni delle librerie per i notebook su GPU (14-16)
```

## Requisiti
Il lavoro è stato eseguito in due ambienti. Gli script `.py` (da `00` a `13` e da `17` a `19`) girano in locale, su CPU
(MacBook con 8 GB di RAM), con Python 3.11 e le librerie in `requirements.txt` (`pip install -r requirements.txt`);
in questa configurazione la cross-validazione di RoBERTa richiede circa 35 minuti per fold. I notebook (da `14` a `16`)
girano su GPU, in Google Colab (ricerca) e su Kaggle (modelli finali): Python 3.13, PyTorch 2.11.0, Transformers 4.57.1,
scikit-learn 1.6.1, GPU Tesla T4 (le versioni sono in `requirements_gpu.txt` e nei `manifest.json` delle cartelle dei risultati).
Il modello `roberta-base` viene scaricato da Hugging Face alla prima esecuzione.
Gli script usano percorsi relativi alla cartella del progetto: si lanciano dalla radice.
Ripetendo lo stesso addestramento su CPU o su GPU, o con versioni diverse delle librerie, i risultati possono differire di poco.

## La pipeline, nell'ordine
| Script | Cosa fa |
|---|---|
| `00_estrazione_da_archivio.py AAAA-MM` | Raccolta dati: estrae i thread di r/TrueFilm da un mese del dump Reddit (Academic Torrents) |
| `01b_integra_manuali.py` | Converte i thread raccolti a mano nello stesso formato |
| `01_pulizia_dati.py` | Preprocessing: pulizia, tipo di thread, indicatore lessicale `candidato_framing` → `data/processed/commenti_puliti.csv` (30.035 commenti) |
| `02_campionamento_pilot_gold.py` | Campiona pilot (100) e gold (800), seme 42. **Non rilanciare**: cambierebbe i campioni già annotati |
| `03_accordo_pilot.py`, `04_etichette_finali_pilot.py` | Pilot: accordo tra annotatori ($\kappa$) e etichette finali dopo la revisione |
| `05_accordo_gold.py`, `06_etichette_finali_gold.py` | Gold: accordo su 222 commenti con due annotatori, revisione dei 24 disaccordi, etichette finali |
| `07_supplemento_training_truefilm.py`, `08_supplemento_training_rmovies.py` | Materiale aggiuntivo per il solo training (Arctic Shift, r/TrueFilm e r/movies) |
| `09_split_test_sviluppo_fold.py` | Test (159 commenti del gold, id fissi in `data/annotation/test_id.csv`) e sviluppo (641) in 5 fold stratificati e raggruppati per thread (`StratifiedGroupKFold`, seme 82) |
| `10_riepilogo_numeri.py` | Tutti i numeri del corpus, del pilot e del gold → `data/riepilogo_numeri.json` |
| `11_baseline_tfidf.py` [`--finale`] | Baseline: cross-validazione sullo sviluppo; con `--finale` addestra sul training completo e valuta sul test |
| `12_roberta_kfold_cv.py` | RoBERTa, **configurazione iniziale**: cross-validazione a 5 fold, un seme per fold, salva ogni fold appena finito (riparte da solo) |
| `13_roberta_finale_test.py` | RoBERTa, configurazione iniziale: modello finale (epoche = mediana delle epoche migliori dei fold, `EPOCHE_FINALI`) e valutazione sul test |
| `14_ricerca_transformer_colab.ipynb` | **Ricerca ampliata** su RoBERTa (e DistilRoBERTa, appendice B): 3 velocità di apprendimento (1e-5, 2e-5, 5e-5), 3 semi (42, 123, 2026), 5 fold, 6 epoche (90 addestramenti); scelta di velocità di apprendimento ed epoca comune ai fold → `modelli/ricerca_ampliata/2_transformer_cv/` |
| `15_modelli_finali_kaggle.ipynb` | Ricerca ampliata: addestra i modelli scelti sul training completo (1.068 commenti) e valuta sul test → `3_modelli_finali/` |
| `16_early_stopping_kaggle.ipynb` (e `.py`) | Prova aggiuntiva: RoBERTa con una sola validation (fold 1), fino a 30 epoche e arresto anticipato → `4_early_stopping_fold1/` |
| `17_applicazione_corpus.py` | Applica RoBERTa (modello finale della ricerca ampliata) a tutto il corpus, solo inferenza → `data/processed/corpus_previsioni_roberta.csv` |
| `18_analisi_esplorativa_corpus.py` | Analisi esplorativa: quote previste per tipo di thread, con intervalli bootstrap sui thread → `data/processed/analisi_esplorativa_tipo_thread.json` |
| `19_figure_capitolo4.py` | Rigenera le figure del capitolo 4 dai file dei risultati (ricerca ampliata) → `docs/figure/` |

Due percorsi per i transformer. **Configurazione iniziale** (locale, CPU): `09` → `10` → `11` → `11 --finale` → `12` → `13`
Prima del `13` controllare la mediana delle epoche che stampa il `12` e, se serve, aggiornare
`EPOCHE_FINALI` in cima al `13`. **Ricerca ampliata** (GPU): `14` → `15` (→ `16`). I notebook leggono i dati preparati da
`09` (`dev.csv`, `supplementare.csv`, `test.csv`) e i risultati precedenti dal pacchetto
`modelli/ricerca_ampliata/thesis_ml_kaggle_inputs.zip`, da caricare su Kaggle come Dataset privato (istruzioni in
`docs/ricerca_ampliata/COLAB_README.md` e `KAGGLE_README.md`). I file dei pesi non sono nella cartella (vedi `.gitignore`).

DistilRoBERTa (appendice B della tesi) è in `scripts/appendice_distilroberta/` (`12b` cross-validazione e `13b` modello finale, stessa procedura di `12` e `13`; si lanciano dalla radice). Lo script del controllo su r/movies con i pesi di classe, scritto ma non eseguito per intero, è in `docs/esperimenti_preliminari/`.

## Il protocollo in breve
- **Gold**: 800 commenti annotati con il manuale v1.1, 222 con due annotatori ($\kappa$ = 0,830). Etichette ufficiali in `data/annotation/gold/gold_etichette_finali.csv`: non si toccano.
- **Test**: 159 commenti del gold (22 strumentali), solo gold. **Sviluppo**: gli altri 641, in 5 fold senza thread condivisi.
- **Training di ogni giro**: gold degli altri 4 fold + pilot (83, tolti quelli con thread del test o della validation) + 344 commenti aggiuntivi. Validation e test contengono **solo gold**.
- **Scelte** con la cross-validazione (macro-F1 media più alta), mai con il test. Configurazione iniziale (`11`–`13`): Logistic Regression contro SVM e baseline contro transformer, con $C = 1$ e velocità di apprendimento massima $2 \cdot 10^{-5}$ fissi, epoche = mediana delle epoche migliori dei fold. Ricerca ampliata (`14`–`15`): velocità di apprendimento ed epoca di RoBERTa insieme, con la media su 5 fold e 3 semi e la stessa epoca in tutti i fold (a parità, meno epoche e velocità più bassa). In entrambe: tenere r/movies con la regola $\delta \ge -\sigma$ (controllo svolto in `docs/esperimenti_preliminari/`).
- **Semi**: nella ricerca ampliata 42, 123 e 2026 per misurare la variabilità (mai per scegliere il migliore); il modello finale usa il seme 42, fissato prima della valutazione. Nella configurazione iniziale: 42 (nella cross-validazione, 42 più il numero del fold).
- **Nota sul test**: il test è stato consultato anche nella fase preliminare del lavoro, con un altro protocollo (vedi `docs/esperimenti_preliminari/LEGGIMI.txt`). Inoltre è stato valutato con due configurazioni di ciascun transformer, quella iniziale e quella della ricerca ampliata: entrambi i risultati sono riportati. I file di risultati riportano la stessa nota.

## Risultati principali (macro-F1)
| | Cross-validazione (media 5 fold) | Test (159 commenti) |
|---|---|---|
| Logistic Regression ($C$ = 1) | 0,554 | 0,604 (IC 95% 0,512–0,692) |
| SVM lineare ($C$ = 1) | 0,538 | – |
| DistilRoBERTa, configurazione iniziale | 0,651 | 0,649 (0,557–0,732) |
| DistilRoBERTa, ricerca ampliata | 0,638 | 0,625 (0,538–0,705) |
| RoBERTa, configurazione iniziale (lr 2e-5, 4 epoche) | 0,685 | 0,714 (0,634–0,786) |
| **RoBERTa, ricerca ampliata (lr 5e-5, 6 epoche)** | **0,679** | **0,771** (0,684–0,840) |

La media di cross-validazione della configurazione iniziale è la media dei massimi per fold (ottimistica); quella della ricerca ampliata usa la stessa epoca in tutti i fold e la media su tre semi (deviazione tra semi 0,019 per RoBERTa).
Tra le due configurazioni di RoBERTa sul test la differenza non è significativa (+0,057, bootstrap appaiato 95%: da −0,025 a 0,136; McNemar esatto $p$ = 0,23).
I commenti dello stesso thread non sono indipendenti: ricampionando i 71 thread del test invece dei commenti gli intervalli sono un po' più ampi (RoBERTa ricerca ampliata 0,676–0,831; baseline 0,475–0,695; differenza RoBERTa − baseline 0,066–0,287, che esclude ancora lo zero; differenza tra le due configurazioni di RoBERTa da −0,033 a 0,121).

## Quale modello ha prodotto i risultati della tesi
Il RoBERTa della **ricerca ampliata** (velocità di apprendimento 5e-5, 6 epoche, seme 42, addestrato su Kaggle sui 1.068 commenti di training): `modelli/ricerca_ampliata/3_modelli_finali/roberta/checkpoint/` (SHA-256 del file dei pesi `ba272a4355074e6d9fb739190214a16db7115aa415956d56d200b10d6792a480`, non incluso nella cartella). Con questo checkpoint sono ottenute le predizioni sul test e quelle sul corpus (riapplicato ai 159 commenti del test riproduce le predizioni di `test_predictions.csv`: 159 su 159). Il RoBERTa della configurazione iniziale è in `modelli/configurazione_iniziale/roberta/`.

File: `modelli/configurazione_iniziale/` (baseline, RoBERTa e DistilRoBERTa nella prima versione dell'esperimento) e `modelli/ricerca_ampliata/` (l'esperimento in più: ricerca di $C$, velocità di apprendimento, semi ed epoche, modelli finali e prova con più epoche; ha un suo `LEGGIMI.md`).

## Dati (`data/`)
- `sorgenti_json/corpus/`: un file JSON per thread, il corpus grezzo. `sorgenti_json/manuali/`: thread raccolti a mano e materiale aggiuntivo.
- `processed/commenti_puliti.csv`: corpus pulito (30.035 commenti: 29.106 dall'archivio in 1.254 thread, 929 da 11 thread scelti a mano).
- `processed/corpus_previsioni_roberta.csv`: previsioni di RoBERTa (ricerca ampliata) su tutto il corpus, senza il testo (si ottiene con `17`). I file `..._iniziale` sono le previsioni e l'analisi ottenute con la configurazione iniziale.
- `annotation/`: `pilot/` e `gold/` hanno la stessa struttura (`strumento/` HTML di annotazione, `risposte_originali/`, `revisione_disaccordi/`, etichette finali e metriche), `extra_training/` il materiale aggiuntivo (con `primo_tentativo_80/`, i primi 80 commenti cercati per parole, scartati dal training), `test_id.csv` gli id del test.
- I commenti vengono da Reddit (r/TrueFilm e, per il solo materiale aggiuntivo, r/movies) e sono testi pubblici.

## Modelli (`modelli/`)
- scritti da `09`: `dev.csv` (641 gold con la colonna `fold`), `supplementare.csv`, `test.csv`, `_split_info.json`.
- `configurazione_iniziale/baseline/`, `roberta/` e `distilroberta/`. La baseline: `baseline_cv.json` (macro-F1 per fold), `baseline_valutazione.json`, `baseline_predizioni_test.csv`, matrice di confusione (la ricerca ampliata ha ripetuto la baseline con tre valori di $C$ e dà le stesse predizioni sul test). RoBERTa e DistilRoBERTa: cross-validazione (`*_cv.json`, macro-F1 per fold e per epoca, e il log di RoBERTa), modello finale e valutazione sul test (`*_valutazione.json`, `*_predizioni_test.csv`, matrice di confusione); per RoBERTa anche il checkpoint (circa 480 MB).
- `ricerca_ampliata/`: risultati dei notebook `14`–`16` in tre cartelle numerate (`2_transformer_cv/`, `3_modelli_finali/`, `4_early_stopping_fold1/`); `1_baseline_C/` contiene un controllo supplementare sulla baseline (valori di $C$), con il suo notebook, non riportato nella tesi e il pacchetto di input per Kaggle. Il `LEGGIMI.md` interno dice quale file aprire per ogni numero della tesi. I file dei pesi (`*.safetensors`) non sono inclusi.

## Documenti (`docs/`)
- `manuale_annotatore.pdf` (v1.0, usata per il pilot), `manuale_annotatore_v1_1.pdf` (usata per il gold) e `manuale_annotatore_v1_1_tesi.pdf` (la stessa versione 1.1, nell'appendice A della tesi).
- `figure/`: solo le figure che compaiono nella tesi. Le due del capitolo 4 (`fig_cv_per_fold`, `fig_matrici_confronto`) si rigenerano con `19` (che scrive anche i PNG, non usati nella tesi); gli schemi (`fig_dati`, `fig_gold_test_sviluppo`, `fig_cv_giri_fascia`) sono disegni. Gli script `03` e `05` salvano in questa cartella altre figure sull'accordo tra annotatori, che non sono usate nella tesi.
- `analisi_errori_roberta_test.csv`: i 33 commenti del test classificati male da RoBERTa (ricerca ampliata), con probabilità, predizione della baseline e categoria proposta. `analisi_errori_configurazione_iniziale.csv`: i 40 errori della configurazione iniziale.
- `esperimenti_preliminari/`: esperimenti del lavoro precedente al protocollo finale (script, risultati e un `LEGGIMI.txt`): documentano che il test è stato usato in quella fase e contengono il controllo su r/movies (macro-F1 media 0,647 con e 0,650 senza) e la traccia della regola fissata prima dei risultati.
- `ricerca_ampliata/`: relazione sulla ricerca ampliata (`PERCORSO_COMPLETO`, in HTML, Word e Markdown), con le sue figure, e le istruzioni per Colab e Kaggle.

## Avvertenze
- Prima di rilanciare `01`, fare una copia di `commenti_puliti.csv`.
- Il gold è definitivo: nessuno script va rilanciato su di esso.
- `docs/esperimenti_preliminari/` documenta le prove del lavoro precedente al protocollo finale (compreso l'uso del test) e il controllo su r/movies; si può ignorare per riprodurre i risultati.
