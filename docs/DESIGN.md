# ModuleFinder design

The method is described page by page on the documentation site
(`docs/source/method/`, published at https://eraslanbas.github.io/ModuleFinder/).
Section numbers below refer to the method paper, which is not part of this
repository.

## Where the code implements what

| paper | code |
|---|---|
| §2 rows, Welch + `ashr` | `module_finder.de` (`ComputeSE.py`, `run_ashr_on_chunk.R`, `shards.py`, `matrices.py`) |
| §3.1 eq. (2) the tied projection `z = x B⁺` | `models.loadings.GlobalLoadings.project` |
| §3.2 eq. (3)–(5) masked projection, Sherman–Morrison | `models.loadings.GlobalLoadings.project`, `data.ontarget.on_target_index` |
| §3.3 eq. (6)–(8) the per-component head | `models.heads.PerComponentHead` |
| §3.4 eq. (9) label network `λ(u)` | `models.label_net.FactorizedLabelNet` |
| §4.1 eq. (9) `L_disc` | `objectives.contrastive.contrastive_loss` |
| §4.2 eq. (10) `L_recon`, precision weights | `objectives.reconstruction`, `GlobalLoadings.squared_error` |
| §4.3 eq. (11) optional TC term | `objectives.total_correlation` (off by default) |
| §6 stratified negatives | `objectives.contrastive.StratifiedNegativeSampler` |
| §7 reading `z`, sign anchoring | `interpret.loadings`, `interpret.effects` |
| §8 eq. (14) the span test | `evaluation.span` |
| Q4 cross-seed stability | `evaluation.stability` (Hungarian + MCC) |
| the training loop, eq. (12) | `train.fit` |

## Deliberately absent

`models/encoder.py` and `models/decoder.py` were removed. `B` is the only
factor→gene map and the projection is derived from it (§3.1) — with a separate
encoder there are two maps that can disagree about what factor `k` means, and
encoder weights are filters rather than patterns. There is also no label prior
`f(e_p, e_c) → z`: it exists only to predict without `x`, and prediction is not
the objective.

## Running it

```bash
python scripts/fit_model.py --contexts Stattic DG-172 --epochs 50 --seed 0
python scripts/fit_model.py --contexts Stattic DG-172 --epochs 50 --seed 1
# then compare the two B matrices with evaluation.stability.match_factors
```

Ablations that the paper argues should degrade the result, available as flags:
`--unconstrained-head` (removes identifiability), `--no-mask` (lets knockdown
efficiency claim a factor), `--tc-weight` (adds independence pressure, which is
not identifiability).
