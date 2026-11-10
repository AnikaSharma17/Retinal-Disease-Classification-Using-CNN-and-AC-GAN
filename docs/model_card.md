# Model card: retinal fundus systemic-disease classifier (research prototype)

## Intended use
Research and education on whether colour fundus photographs carry signal for systemic
conditions (diabetes mellitus, hypertension, heart disease, hypercholesterolaemia).
**Not** for screening, diagnosis or any clinical decision.

## Model
* Primary: EfficientNet-B0 (ImageNet-pretrained), global-average-pooled, sigmoid head over
  four labels (multi-label: a patient can have several conditions); head trained first,
  then the top 20 backbone layers fine-tuned with BatchNorm frozen.
* Baseline: the original project CNN (3 conv blocks) trained from scratch.
* Input: RGB fundus photo, resized to 224x224, luminance histogram-equalized.

## Data
* Source: a clinician-provided set of fundus photographs (origin, consent and acquisition
  details are not documented in this repository; they must be before any publication).
* The repository copy contains **38 unique source images**. The other 3,268 files are
  offline-augmented copies that were split after augmentation (leakage) and are excluded.
* 11 of the 38 images carry more than one disease label; there is a single healthy control.
* Patient identifiers are not available, so two images of the same patient could fall in
  different splits. Group by patient with `--patients` once IDs exist.

## Evaluation
See `README.md#results` for the measured numbers with bootstrap 95% confidence intervals
and baselines. With a held-out test split of a handful of images, every metric has a CI
spanning most of [0, 1]; **no performance claim is supported by the current data**.

## Known limitations and risks
* Sample size far below what systemic-disease prediction from fundus images requires
  (published work uses tens of thousands of patients).
* Label semantics unclear (comorbidity vs. filing error) for the multi-label images.
* Severe class imbalance (DM dominates; one control).
* Single source / camera: no evidence of generalization to other devices or populations.
* Grad-CAM is a coarse, post-hoc explanation; a plausible heatmap is not proof of a
  correct reason.

## Ethical considerations
Predictions about systemic disease are sensitive health information. Any deployment would
need clinical validation, regulatory review, consent for secondary data use, and subgroup
(age, sex, ethnicity, camera) performance analysis, none of which exists today.
