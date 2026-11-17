# CIFM-XAI

CIFM-XAI combines projected features, Isolation Forest anomaly scores, validation-weighted ensemble fusion, and permutation importance. The notebook reproduces separate training and held-out evaluations on SeeGULL Multilingual and Cross-Cultural Communication. It uses grouped partitions and five random seeds.

## Files

- `CIFM-XAI.ipynb` — Google Colab notebook for preprocessing, training, evaluation, and result export.

## Datasets

- [SeeGULL Multilingual](https://github.com/google-research-datasets/SeeGULL-Multilingual)
- [Cross-Cultural Communication Interaction Dataset](https://www.kaggle.com/datasets/zara2099/cross-cultural-communication-interaction-dataset)

Download the CSVs from the original sources and place them in a ZIP archive named `Strategic_Identity_Fragmentation_CIFM.zip`. The notebook expects this internal directory structure.

```text
Strategic_Identity_Fragmentation_CIFM/
└── data/
    └── raw/
        ├── seegull/
        │   └── SeeGULL-Multilingual-main/
        │       └── dataset/
        │           ├── offensiveness/   (23 CSVs)
        │           └── stereotypes/     (23 CSVs)
        └── communication/
            └── cross_cultural_communication_dataset.csv
```

Keep the original SeeGULL CSV filenames and column names. The notebook reads the files directly from the ZIP.

## Reproduce the experiments

1. Open `CIFM_XAI_Fast_GPU_One_Cell.ipynb` in Google Colab.
2. Select a GPU runtime. The manuscript experiments used an NVIDIA Tesla T4.
3. Run the notebook cell. Upload `Strategic_Identity_Fragmentation_CIFM.zip` when prompted, or set `ZIP_PATH` to the archive's existing location.
4. Wait for `Finished. Download ...` to appear, then save the generated `CIFM_XAI_results_*.zip` file.

The notebook installs the required packages and downloads pretrained models from Hugging Face. It uses seeds `42`, `127`, `340`, `541`, and `819`. The mBERT and XLM-R baselines are evaluated on SeeGULL.

## Outputs

| File | Contents |
|---|---|
| `results_summary.csv` | Held-out metrics aggregated across seeds, bootstrap confidence intervals, and paired group-level comparisons |
| `metrics_by_seed.csv` | Per-seed metrics and confusion-matrix counts |
| `test_predictions.csv.gz` | Held-out labels, prediction scores, and predicted classes |
| `paired_seed_tests.csv` | Seed-level statistical comparisons |
| `ensemble_weights.csv` | Validation-derived ensemble weights |
| `permutation_importance.csv` | Held-out permutation importance for seed 42 |
| `split_summary.csv`, `split_manifest.csv`, `data_audit.json` | Partition counts, record assignments, and data checks |
| `environment.json` | Hardware, software versions, and run configuration |
| `f1_*.png`, `confusion_*.png` | Held-out F1 comparisons and confusion matrices |

## Contact

For questions about the code or reproducibility, open an issue in this repository.
