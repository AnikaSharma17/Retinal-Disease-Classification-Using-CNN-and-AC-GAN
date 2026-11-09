# Data and model versioning

The repository currently stores ~1.9 GB of images and ~370 MB of `.h5` weights directly in
git (`.git` is ~2 GB). Every clone downloads all of it, and any change to the data rewrites
large blobs. Two options, in order of preference:

## Option A: DVC (recommended for an ML project)

Keeps data out of git, versions it by content hash, and ties each experiment to an exact
data version (`data/manifest.csv` already records a sha256 per image).

```bash
pip install dvc
dvc init
dvc add dataset models/legacy           # writes dataset.dvc, models/legacy.dvc; adds them to .gitignore
git rm -r --cached dataset models/legacy
dvc remote add -d storage <remote>      # s3://..., gdrive://..., or a local/NAS path
dvc push
git add dataset.dvc models/legacy.dvc .dvc .gitignore
git commit -m "Track images and legacy weights with DVC"
```

## Option B: Git LFS (simplest if you stay on GitHub)

```bash
git lfs install
git lfs track "*.jpg" "*.jpeg" "*.png" "*.h5" "*.keras"
git add .gitattributes
git add --renormalize .
git commit -m "Move images and weights to Git LFS"
```

GitHub's free LFS quota is 1 GB storage / 1 GB bandwidth per month, which this dataset
exceeds.

## Shrinking existing history (optional, destructive)

Both options only stop *new* growth. To remove the blobs from history, rewrite it with
`git filter-repo` (or `git lfs migrate import --everything`) and **force-push**. That
changes every commit hash and breaks existing clones/forks, so do it deliberately and
only after a backup.

## Patient-level metadata

To make splits patient-safe, provide `patients.csv` (`filename,patient_id`) and rebuild:

```bash
python -m src.data.manifest build <originals_dir> --out data/manifest.csv --patients patients.csv
python -m src.data.manifest split data/manifest.csv --out data/manifest_split.csv --seed 42
```

Commit `data/manifest*.csv`. They are small, human-reviewable, and define the dataset.
