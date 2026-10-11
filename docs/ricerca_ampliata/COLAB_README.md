# Run the thesis experiments in Colab

Use these two standalone notebooks. They include their own code; the original Python scripts do not need to be uploaded.

1. `01_baselines_colab.ipynb`: TF-IDF with logistic regression and linear SVM, comparing `C = 0.1, 1, 10` on five folds. A CPU runtime is sufficient. It also trains and evaluates the selected final baseline.
2. `02_transformers_colab.ipynb`: RoBERTa and DistilRoBERTa, comparing learning rates `0.00001, 0.00002, 0.00005`, three training seeds and five folds. Choose a GPU runtime. It also trains and evaluates the selected final transformers.

## Data to put in Google Drive

Create `MyDrive/thesis_ml/data/` and upload these prepared files from the original project's `modelli/` folder:

```text
dev.csv
supplementare.csv
test.csv
```

These are the outputs of `09_split_test_sviluppo_fold.py`. They contain the cleaned text, final labels and existing fold assignments. The copied scripts in this workspace do not include these data files. If they have not been generated, obtain them from the original project before running the experiments.

`dev.csv` needs `id`, `thread_id`, `testo`, `etichetta`, `fold`. The other two need the same columns except `fold`. The notebooks expect 641 development comments and 159 test comments, and the labels `strumentale`, `intrinseco`, `ambiguo`.

If you prefer another Drive folder, edit `PROJECT_DIR` in both notebooks.

## Running the notebooks

Open [Google Colab](https://colab.research.google.com/), select **File → Upload notebook**, and upload notebook 01. Run the install cell first in a fresh runtime, mount Drive, check the paths, then run the development search and summary cells.

Upload notebook 02 and select **Runtime → Change runtime type → GPU**. Run the cells in order. The complete transformer grid is 90 training runs, each covering six epochs. By default, a session completes up to 10 new runs. Rerun the search cell or reconnect later to continue. `ACTIVE_MODELS` lets you choose one model for a session while keeping the full experiment unchanged.

Completed runs are stored under `MyDrive/thesis_ml/results/` and skipped on rerun. If Colab disconnects during a fold, that unfinished fold restarts; previously completed runs remain saved. Use one session at a time for an experiment folder. Colab availability and session duration vary; the grid may require multiple sessions.

Changing the grid, input data, training code, model revision or software versions creates a separate experiment folder. A different GPU type can resume an experiment, but is recorded because it can add numerical variation. Keep the environment consistent where possible.

## Selection and final evaluation

Both notebooks initially set `RUN_FINAL = False`. The search does not load `test.csv`.

The baseline search selects regularization by mean fold macro-F1. The transformer search selects a learning-rate/epoch pair by mean macro-F1 across the same five folds and all seeds. It chooses one common epoch for a configuration. Fold and seed variability are reported separately.

Once both searches are complete, freeze the comparison set: the best baseline and the selected configuration of each transformer. Set `RUN_FINAL = True`, rerun the configuration cell and run the final section in each notebook. New models train on all development plus supplementary examples. Their final test predictions, metrics and model files are saved in Drive.

The learning-rate schedule still covers six epochs when a selected final model stops earlier, matching the search procedure. The final training seed is fixed beforehand at 42.

The original test had already been used in preliminary experiments. Preserve that limitation in the thesis; this workflow cannot make it an untouched holdout retrospectively. The winning CV scores are development scores used for selection, not unbiased final performance estimates. Do not use the final test results to tune again.

## Why these parameters?

The transformers are fine-tuned from pretrained weights. Small learning rates make controlled updates to those weights. In decimal notation the values look similar, but the original `1e-5, 2e-5, 3e-5` grid spans a factor of three. The new grid spans a factor of five. The original [RoBERTa paper](https://arxiv.org/abs/1907.11692) itself used the narrower grid in its GLUE experiments. These are starting candidates, not known optima for this dataset.

`C` belongs to the logistic regression or SVM classifier, not to TF-IDF. Smaller `C` means stronger regularization: it penalizes large weights more strongly. Larger `C` means weaker regularization and allows a closer fit to training data. Cross-validation checks which balance works better. See the [logistic regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [linear SVM documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

## Validation performed here

Notebook structure and Python cells were checked. The full 30-fit baseline grid, selection, resume behavior and final inference were exercised on synthetic data. Transformer training and final checkpoint saving were exercised with a tiny locally initialized RoBERTa model, including seed repeatability, common-epoch selection, incomplete-grid rejection and resume behavior.

The real thesis data, pretrained RoBERTa experiments and Colab GPU runtime have not been executed here. Those runs occur in Colab.
