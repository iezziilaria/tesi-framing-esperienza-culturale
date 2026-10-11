# Final training on Kaggle

Use `03_final_models_kaggle.ipynb` and `thesis_ml_kaggle_inputs.zip`. The original Colab notebooks remain unchanged.

For the subsequent longer-training experiment, use `04_roberta_early_stopping_kaggle.ipynb` with the same already-uploaded Dataset. Alternatively, first run the package-install cell from that notebook, then copy all of `04_roberta_early_stopping_kaggle.py` into one Kaggle code cell. It performs one fresh RoBERTa fit with fold 1 as validation, a 30-epoch maximum, and patience 5 monitoring validation macro-F1. This split has 938 training rows and 123 validation rows; seven supplementary rows sharing validation threads are excluded. Its `best_checkpoint/`, validation reports and curves go into a separate `roberta_holdout_<fingerprint>` directory. The test file is never read. Save the notebook version with outputs after completion. Change the `CONFIG` dictionary to adjust the epoch cap or patience. The scheduler spans the new epoch cap.

Notebook 04 was checked against the real uploaded package without accessing the test file, then its training and checkpoint logic were exercised on a tiny local model and synthetic data. A controlled score sequence verified patience, earlier-epoch tie handling, and that the saved weights come from the best epoch rather than the last epoch.

## Upload and run

1. On Kaggle, go to **Datasets → New Dataset**. Upload `thesis_ml_kaggle_inputs.zip`, give it a name such as `thesis-ml-final`, and keep it **private**. Kaggle extracts ZIP contents and preserves directory structure. [Dataset documentation](https://www.kaggle.com/docs/datasets).
2. Create a notebook, then use **File → Import Notebook** to import `03_final_models_kaggle.ipynb`.
3. Select **Add Input** and attach the dataset created in step 1. [Notebook documentation](https://www.kaggle.com/docs/notebooks).
4. In notebook settings/session options, enable a **GPU** accelerator and **Internet**. The notebook uses one GPU. Internet is needed for dependency installation and pretrained model downloads.
5. Run cells in order. The notebook detects `thesis_ml/bundle_manifest.json` under `/kaggle/input`, checks the uploaded files, and reads the frozen selections automatically. No path edits or Google Drive mount are required. If more than one thesis package is attached, set `PACKAGE_DIR` to the directory containing its `bundle_manifest.json`.
6. After training, use **Save Version → Quick Save**, with output saving enabled in the save options, then confirm that the checkpoint files appear in the saved version's **Output** tab. Download the experiment folder before ending the session. A fresh **Save & Run All** performs another execution; use it deliberately rather than to archive an already completed interactive run. [Kaggle's save-options announcement](https://www.kaggle.com/product-feedback/161328).

## What it trains

| Model | Frozen setting |
| --- | --- |
| TF-IDF + logistic regression | C=1, balanced class weights |
| RoBERTa | LR=5e-5, 6 epochs |
| DistilRoBERTa | LR=2e-5, 5 epochs |

The final seed is 42. Both transformers retain the original six-epoch scheduler horizon, batch size, text limit, weighted loss, and pinned pretrained model revisions. The notebook trains on all development plus supplementary examples, checks that test IDs and threads do not overlap, and evaluates each model on the same test set. It does not run folds or select settings from test scores.

Outputs go to `/kaggle/working/thesis_ml/results/kaggle_final_<fingerprint>/`. They include the original selection provenance, Kaggle environment, checkpoints, predictions, class metrics, confusion matrices, and `final_comparison.csv`. Environment changes create a new final experiment rather than overwriting the Colab CV identity.

`RUN_FINAL=True` by default. Set it to False for inspection only. `ACTIVE_MODELS` can restrict a session to one transformer; `RUN_BASELINE` controls the selected baseline fit. Completed models are skipped when rerunning their cells in the same final experiment. Interrupted model training restarts. For a fresh session, attach saved notebook outputs through **Add Input** and copy the previous `kaggle_final_<fingerprint>` folder into `/kaggle/working/thesis_ml/results/` before the provenance cell. Resume requires matching data, settings, code and recorded environment.

## Before final training

The existing 90-run search supports final training now. A longer CV experiment with a 10–12 epoch cap is optional because RoBERTa's best score lies at epoch six; its increase from epoch five is small. If you choose to extend the cap, finish that new CV experiment and regenerate the selections before using this package. Do not simply extend final training based on test scores. This package uses the current six-epoch selection.

This notebook performs one final fit per model with the prespecified seed. Repeating final fits over the three prespecified seeds is optional if final-training variability is part of the thesis analysis; report all runs or their mean and spread rather than choosing the seed with the best test score. The existing test set was used in preliminary experiments, so describe that limitation when reporting final results.

## Local verification

The notebook passed schema and syntax validation, package/hash checks, frozen-selection loading, and a check that setup never reads test labels. Its actual final-training loop was exercised using synthetic data and two tiny local RoBERTa models, including selected epochs/rates/revisions, checkpoint reload, completed-model skipping, and input-change rejection. The supplied test set was not evaluated locally, and full-size GPU training still takes place on Kaggle.
