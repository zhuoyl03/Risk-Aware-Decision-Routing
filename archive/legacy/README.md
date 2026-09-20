# Original experiment scripts

These scripts record the training and evaluation setup used before the routing
package. Their contents are preserved unchanged, including the two-stage ViT
training code associated with the local stage1/stage2 checkpoints.

Use `src/radr` for current experiments. The historical scripts have limitations:

- Some command-line arguments are parsed but not applied.
- Unreadable images can be skipped without a complete exclusion manifest.
- The old evaluation compares review selections with ground-truth severity;
  this is an oracle-style analysis, not an available deployment signal.
- Its `severity_error_rate` does not depend on the selected review cases.

These files explain the checkpoint's history. Current metrics are produced by
the tested routing package and should not be compared directly with the old
deferral metrics.
