# Results report

All numbers are on the held-out **test** split unless noted; brackets are percentile-bootstrap 95% CIs. `n/a` means the metric is undefined for the available labels; † means n < 10, too few samples for a meaningful CI (the point estimate is essentially noise).

## multilabel

| model | n | auc_macro | macro_f1 | subset_accuracy |
|---|---|---|---|---|
| baseline: prior | 5 | 0.50† | 0.22† | 0.40† |
| baseline: colour_lr | 5 | 0.42† | 0.29† | 0.00† |
| baseline: embedding_lr | 5 | 0.58† | 0.31† | 0.40† |
| multilabel_efficientnet | 5 | 0.42† | 0.22† | 0.40† |
| multilabel_efficientnet_cv5 | 5 | 0.42† | 0.22† | 0.40† |
| multilabel_efficientnet_cv5 (5-fold out-of-fold, train+val) | 33 | 0.56 [0.45, 0.67] | 0.25† | 0.45† |
| multilabel_small_cnn | 5 | 0.61† | 0.00† | 0.00† |

## multiclass

| model | n | accuracy | balanced_accuracy | macro_f1 | auc_macro_ovr |
|---|---|---|---|---|---|
| baseline: prior | 3 | 0.67† | 0.50† | 0.40† | 0.50† |
| baseline: colour_lr | 3 | 0.33† | 0.25† | 0.25† | 0.50† |
| baseline: embedding_lr | 3 | 0.67† | 0.50† | 0.40† | 0.50† |
| multiclass_dm_vs_hypertension | 3 | 0.67† | 0.50† | 0.40† | 1.00† |

- `multiclass_dm_vs_hypertension` split counts: train {'DM': 12, 'Hypertension': 4}, val {'DM': 2, 'Hypertension': 1}, test {'DM': 2, 'Hypertension': 1}
- `multilabel_efficientnet` split counts: train {'DM': 19, 'Hypertension': 8, 'Heart Disease': 4, 'Cholesterol': 4, 'Control (all-negative)': 1}, val {'DM': 4, 'Hypertension': 2, 'Heart Disease': 1, 'Cholesterol': 0, 'Control (all-negative)': 0}, test {'DM': 4, 'Hypertension': 2, 'Heart Disease': 1, 'Cholesterol': 0, 'Control (all-negative)': 0}
- `multilabel_efficientnet_cv5` split counts: train {'DM': 19, 'Hypertension': 8, 'Heart Disease': 4, 'Cholesterol': 4, 'Control (all-negative)': 1}, val {'DM': 4, 'Hypertension': 2, 'Heart Disease': 1, 'Cholesterol': 0, 'Control (all-negative)': 0}, test {'DM': 4, 'Hypertension': 2, 'Heart Disease': 1, 'Cholesterol': 0, 'Control (all-negative)': 0}
- `multilabel_small_cnn` split counts: train {'DM': 19, 'Hypertension': 8, 'Heart Disease': 4, 'Cholesterol': 4, 'Control (all-negative)': 1}, val {'DM': 4, 'Hypertension': 2, 'Heart Disease': 1, 'Cholesterol': 0, 'Control (all-negative)': 0}, test {'DM': 4, 'Hypertension': 2, 'Heart Disease': 1, 'Cholesterol': 0, 'Control (all-negative)': 0}

## Grad-CAM shortcut check

Share of Grad-CAM mass outside the retinal disc (0 = all attention on the retina).

- `multilabel_efficientnet` (test, n=5): mean 0.26
- `multilabel_efficientnet` (val, n=5): mean 0.27

## AC-GAN sample quality (train split, 64x64)

| class | n real | n fake | KID (mean ± std) | FID |
|---|---|---|---|---|
| DM | 12 | 50 | 0.465 ± 0.004 | 323 |
| Hypertension | 4 | 50 | 0.477 ± 0.017 | 360 |
| pooled | 16 | 100 | 0.461 ± 0.006 | 327 |

KID/FID are on InceptionV3 features; lower is better. FID is strongly biased at this n.

## Ablation: ablation_augmentation

Test metrics, mean ± std over seeds.

| variant | seeds | n test | auc_macro | macro_f1 | subset_accuracy |
|---|---|---|---|---|---|
| no_augmentation | 3 | 5 | 0.556 ± 0.048 | 0.389 ± 0.000 | 0.600 ± 0.000 |
| classic_augmentation | 3 | 5 | 0.611 ± 0.048 | 0.278 ± 0.096 | 0.467 ± 0.115 |
| no_histogram_equalization | 3 | 5 | 0.528 ± 0.048 | 0.231 ± 0.016 | 0.200 ± 0.000 |

## Ablation: ablation_gan

Test metrics, mean ± std over seeds.

| variant | seeds | n test | accuracy | balanced_accuracy | macro_f1 | auc_macro_ovr |
|---|---|---|---|---|---|---|
| real_only | 3 | 3 | 0.667 ± 0.000 | 0.500 ± 0.000 | 0.400 ± 0.000 | 0.500 ± 0.000 |
| real_plus_gan_10 | 3 | 3 | 0.667 ± 0.000 | 0.500 ± 0.000 | 0.400 ± 0.000 | 0.500 ± 0.000 |
| real_plus_gan_40 | 3 | 3 | 0.667 ± 0.000 | 0.500 ± 0.000 | 0.400 ± 0.000 | 0.667 ± 0.289 |
