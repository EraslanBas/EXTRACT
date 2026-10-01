# Reading the factors

## Loadings are already in logFC units

Row $\mathbf B_k$ is factor $k$'s gene program. Because the model is fitted on
unscaled shrunken logFC, $\mathbf B$ needs no back-transform: a loading is in
log-fold-change units per unit of activation. Loadings are always read off
$\mathbf B$, never off a projection's weights, which are filters rather than
patterns.

## Two signed quantities

$z_k$ and $\psi_k$ are both signed and mean unrelated things.

- **$z_k$** is the activation of factor $k$ for one row, the coefficient in
  $\mathbf x \approx \sum_k z_k \mathbf B_k$. Its sign is the direction the
  program moves. If $\mathbf B_k$ loads positively on cholesterol-biosynthesis
  genes, $z_k > 0$ is up-regulation of that program, $z_k < 0$
  down-regulation, and $z_k \approx 0$ means the perturbation leaves it alone.
  **"Does not regulate" is $|z_k| \approx 0$, not $z_k < 0$.**
- **$\psi_k$** is an evidence contribution inside the head; its sign says
  whether that factor supports or contradicts the pairing of the row with its
  label.

Zero is the natural reference for $z$: $\mathbf x$ is already a difference
against within-context controls, so $z_k$ is a change in program activity
relative to unperturbed cells, and zero means "no change".

## Sign and scale must be anchored

$z_k$ and $\mathbf B_k$ can be flipped together without changing the model,
since $(-z_k)(-\mathbf B_k) = z_k\mathbf B_k$, and per-factor scale is free in
the same way. Fix the sign by a convention (the largest-magnitude loading is
positive), or better by anchoring against a perturbation whose direction is
known. Only then does "perturbation $p$ suppresses program 7" mean anything.

Magnitude needs no anchoring: $|z_k|$ ranks how strongly each $(p, c)$ engages
a program, and the ranking is invariant to the flip. Relative loadings *within*
a factor are meaningful; magnitudes *across* factors are not, so rank or
standardise within a factor before comparing. For enrichment, run the two tails
of each factor separately: an unanchored single list conflates genes the
factor raises with genes it lowers.

## Perturbation, context, interaction

A two-way decomposition of the activations separates the three effects of
interest:

```{math}
z_{pc} = \bar z + \mu_p + \gamma_c + \delta_{pc},
```

- $\mu_p$, the **perturbation** main effect: which programs a perturbation
  moves, regardless of context;
- $\gamma_c$, the **context** main effect;
- $\delta_{pc}$, the **interaction**: where a perturbation's effect on a
  program depends on the context. This is the scientific payload of a
  multi-context screen, and it is identically zero with one context.

$\delta_{pc}$ *is* epistasis by construction: since $\mathbf x$ is a difference
against within-context controls, pure additivity would give the same logFC for
$p$ in every context and hence $\delta_{pc} = 0$. All three terms live in the
same factor space and are read through the same $\mathbf B$. The decomposition
is estimated after fitting from the recovered $\mathbf z$; it is not a model
component.

## Which programs are responsive in a context

A perturbation can only shift a program that its context leaves operative. If
context $c$ has silenced program $k$, or saturated it, no perturbation in the
panel can move it, and $z_k$ will not vary across perturbations within $c$. The
spread across the panel,

```{math}
a_{kc} = \operatorname{Var}_p\big[z_k(p, c)\big],
```

is therefore a readout of how modulable program $k$ is in context $c$:

| | reading |
|---|---|
| $a_{kc}$ large | the program is live and the panel engages it |
| $a_{kc} \approx 0$, $\gamma_{kc}$ far from 0 | the context moved the program and locked it there: saturation or a hard block |
| $a_{kc} \approx 0$, $\gamma_{kc} \approx 0$ | nothing in the panel engages the program in this context |

$a_{kc}$ inherits the per-factor scale freedom, so compare it across contexts
for a fixed $k$, not across $k$. It is a variance and so is inflated by
estimation noise; compare contexts on full-data rows, or carry the precision
weights into the variance.
