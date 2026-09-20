# Reproducing the experiments

The repository contains the implementation, tests, measured result tables and
figures. Datasets, weights and feature caches stay outside version control.

## What a fresh checkout can do

Install Python 3.10+ and the package with `python -m pip install -e .`.
Then run `python -m unittest discover -s tests -v`. These tests need no model
download or MEDIC images. The committed figures and result tables can be viewed
without installing anything.

For CUDA, install a compatible PyTorch/torchvision pair for your environment.
[requirements-tested.txt](../requirements-tested.txt) records the September 18
environment, including CUDA wheel suffixes; it is not a portable CPU lockfile.

## Required local artifacts

```text
data/
  MEDIC_train.tsv
  MEDIC_dev.tsv
  MEDIC_test.tsv
  data/                       image paths referenced by the TSV files
out/models/
  stage2_best.pth              original two-class ViT state
```

Obtain MEDIC through its authors' distribution and follow the dataset's terms.
The original fine-tuned checkpoint is not bundled or downloadable from this
repository. Without it, the exact reported experiment cannot be reproduced.
The [historical scripts](../archive/legacy/README.md) document its training
context but have known limitations; backbone retraining was not validated in
this release.

The extractor expects a ViT-base patch16/224 two-class checkpoint with a
`model` state dictionary. Its SHA-256 is recorded in
[provenance.json](../reports/provenance.json). Labels are
`informative=0`, `not_informative=1`.

## Full and cached runs

From the repository root, in the installed environment:

```bash
bash scripts/reproduce.sh --full
```

This runs tests, audits image hashes, extracts image features, trains the small
heads and routers, runs corruption experiments, and renders the report. Logs
and trained artifacts are written under `out/`; report files under `reports/`
are regenerated. Copy report files elsewhere first if you want to preserve a
different run.

After extraction:

```bash
bash scripts/reproduce.sh --cached
```

The cached run refits heads/routers and redraws the report. It does not repeat
image corruption experiments. The report checks the feature-cache fingerprint
to reject corruption results from a different image cache.

Use `RADR_PYTHON=/path/to/python bash scripts/reproduce.sh --full` to select an
interpreter. The runner sets its source path and CPU thread counts itself.

Individual steps are available as modules:

```bash
python -m radr.extract --data-dir data --checkpoint out/models/stage2_best.pth
python -m radr.experiment
python -m radr.stress --samples 3000
python -m radr.report
```

## Reading and using the outputs

- `out/features/`: audited manifests, exclusion counts, features, logits.
- `out/routing/`: fitted router bundles, training histories and test decisions.
- `reports/routing_curves.csv`: every seed, policy and budget.
- `reports/provenance.json`: experiment settings and source/cache fingerprints.
- `reports/index.html`: standalone local report; open it in a browser.

For a routing demonstration:

```bash
radr-route --feature-cache out/features/test.npz --limit 20
radr-route --images image-a.jpg image-b.jpg image-c.jpg image-d.jpg image-e.jpg
```

The default router is seed 17. Cache input must match its recorded fingerprint.
For new images, use `--images`; the checkpoint fingerprint is checked before
inference. Only load trusted local joblib router bundles.

At a 20% review budget, a five-image batch gets one slot. Other capacities use
the baseline unless separately audited. Priority is assigned across the entire
supplied batch, not independently to each image.

## Validation scope

The September 18 [validation record](../reports/validation.json) records the
full local experiment and its limitations. The local audit scripts in
`scripts/` also inspect owner-side backups and saved artifacts; they are not
fresh-checkout tests. `unittest discover` is the portable test entry point.

The full experiment used a WSL environment and an RTX 4070 Laptop GPU.
Three seeds repeat router fitting and development partitioning, not backbone
training. TF32 is disabled to keep direct inference consistent with extraction.
Numerical results on other software/hardware are not promised to be bit-identical.
