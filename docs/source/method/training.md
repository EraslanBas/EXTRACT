# Training and model selection

## Three partitions, split by pair

Rows are split at the level of the **(perturbation, context) pair**, never at
the level of individual rows. All rows of a pair (full-data and subsamples)
share the same cells, so a row-level split leaks and every number computed from
it is meaningless.

| partition | how it is drawn | what it is used for |
|---|---|---|
| **train** | the rest of train/val | fitting |
| **validation** | about 10% of *each perturbation's* train/val pairs, at least one pair always kept in train | choosing the epoch and the hyperparameters; looked at repeatedly |
| **test** | 10% of all pairs, drawn first, from metadata alone; every perturbation keeps at least one train/val pair | read **once**, for the configuration validation chose |

A held-out pair is an unseen combination of a seen perturbation and a seen
context, which is what the model has to supply: the interaction. Holding out
whole perturbations is not a prediction test, because an unseen perturbation's
embedding $e_p$ is never trained.

Everything derived from values (the gene list, which genes count as moved) is
computed from train/val rows only. The negative sampler is built on training
rows, so no validation or test pair is ever named, even as a negative.

**Synthetic rows** are made by shuffling each gene's values across all rows of
a context, before the split, and each synthetic row goes to the partition of
the label it carries.

## Hyperparameters

| | meaning | role |
|---|---|---|
| $d$ | number of factors (rows of $\mathbf B$) | capacity: how many programs |
| $\alpha$ | weight of reconstruction against discrimination | trades how well $\mathbf B$ describes the responses against how well it separates labels |
| $\beta$ | weight of the negative reconstruction term on synthetic rows | how strongly the subspace is pushed off the shuffled background |
| $K$ | subsample rows per pair kept in training, plus the full-data row | how many re-estimates of each pair the model sees |

$K$ selects a **nested** random subset of each pair's ten subsamples, fixed per
pair ($K = 1 \subset K = 2 \subset \dots \subset K = 10$), so a difference
between two values of $K$ is due to $K$ and not to which subsamples were drawn.
$K$ thins the **training** rows only; validation and test keep all their rows,
so every $K$ is scored on the same rows.

## Choosing the epoch

Every model trains for a fixed number of epochs, the same for all, and is
evaluated on validation at regular intervals. The kept state is the epoch with
the **highest validation discrimination accuracy**: each validation row scored
under its own label and under one permuted label, with chance at 0.5.

Accuracy rather than validation loss decides the epoch. As the head grows
confident, validation cross-entropy rises from early in training even while
validation accuracy keeps improving, because the loss punishes the few
confidently wrong rows heavily. Selecting on the loss would keep a nearly
untrained model. Training continues past the selected epoch so the full curves,
and the overfitting onset, are recorded, and $\mathbf B$ is saved at every
evaluation, so the epoch can be re-chosen under another rule without refitting.

## Choosing a configuration

Configurations are compared at each one's selected epoch, on validation only,
using two scale-free numbers:

1. **validation accuracy**, first;
2. **validation reconstruction error on measured rows**, to break ties.

The total loss is not comparable across configurations: its
$-\alpha\beta\,\mathcal L_{\text{recon,synthetic}}$ part falls with
$\alpha\beta$ whatever the model does.

Accuracy differences within noise are not real differences. A difference
smaller than the spread across seeds, or across random draws of the validation
negatives, does not separate two configurations. Among configurations whose
accuracy is indistinguishable from the best, prefer better reconstruction and
then smaller $d$. Seeds are run only after the grid has narrowed the space.

## The single test read

Once a configuration is chosen, its test metrics are read **once**, and the
read is logged. Choosing among configurations by their test scores would make
test a second validation set, and its numbers would no longer be held out. The
gap $|\text{val} - \text{test}|$ is reported, because a large gap is the one
failure a three-way split still admits: selection overfitted to validation.
