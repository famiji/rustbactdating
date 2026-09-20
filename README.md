# rustbact

A Rust reimplementation of the core MCMC engine of
[BactDating](https://github.com/xavierdidelot/BactDating), built for large
phylogenies (tens of thousands of tips) where the R implementation becomes
prohibitively slow.

## Motivation

BactDating's `bactdate()` recomputes the full likelihood over every branch on
each Metropolis-Hastings move, giving roughly `O(n^2)` per-iteration cost. On
trees with tens of thousands of tips that makes a single run take hours, and
since checking convergence requires several independent chains, a full
analysis becomes impractical.

This implementation keeps a running likelihood and updates only the branches
touching the node whose date changed, which is `O(degree(node))` per move.

## Features

- `arc` (default), `carc`, `poisson`, `negbin`, `strictgamma`, `relaxedgamma` likelihoods
- Incremental likelihood updates (validated against full recomputation to 1e-10)
- Multi-chain parallelism (`--chains N`), one thread per chain
- Convergence diagnostics: split-R-hat and ESS, reported per parameter
- Checkpoint / resume: stop and continue a chain without losing samples
- R-compatible distributions (`dgamma`, `pgamma`, `dnbinom`, `dpois`, RNGs),
  unit-tested against R's values to 1e-6

## Install

### 1. Get a Rust toolchain

If `cargo --version` already works, skip this. Otherwise install it with
[rustup](https://rustup.rs) (any 1.70+ toolchain is fine):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# then either open a new shell, or:
source "$HOME/.cargo/env"
```

### 2. Clone and build

```bash
git clone https://github.com/famiji/rustbactdating.git
cd rustbactdating
cargo build --release
```

The binary is written to `target/release/rustbact`. Building takes a minute or
two; there are no dependencies to install beyond the Rust toolchain itself.

### 3. Check it runs

```bash
./target/release/rustbact            # prints usage
cargo test --release                 # 14 tests: distributions, Newick, likelihood invariant
```

Both should complete without errors. `cargo test` finishes in well under a
second once the build is warm.

Then go to **Inputs** to prepare your two files, and **Run** for the actual
analysis commands.

## Inputs

Two files, both plain text. The binary reads them directly, so nothing else
is required to prepare or run the analysis.

1. **`tree.nwk`** — a Newick tree whose branch lengths are in **number of
   substitutions** (not substitutions per site). This is the same unit
   BactDating's `bactdate()` expects, and it matters: `mu` is reported in
   substitutions per year, so a per-site tree leaves the rate short by a
   factor of the alignment length.

   Trees from BEAST, IQ-TREE, RAxML or Gubbins all use this text format;
   only the unit may differ, and most of those programs report substitutions
   per site. Both cases are handled directly by the tool — ask it to check:

   ```bash
   ./target/release/rustbact tree.nwk dates.tsv --check-lengths
   # total branch length: 0.010000
   # warning: this looks like substitutions per site, not substitutions.
   ```

   If it warns, pass the number of alignment sites the tree was built from and
   every branch is scaled before the run:

   ```bash
   ./target/release/rustbact tree.nwk dates.tsv 10000 10 --scale-lengths L
   # tree: tips=... nnode=... total_len=... (scaled xL from 0.095...)
   ```

   The shape of the tree is untouched and the dating result is unchanged — only
   the unit in which `mu` is expressed changes. (The same conversion can be done
   ahead of time with `awk` if you prefer to keep the file itself in substitution
   units.)

2. **`dates.tsv`** — two tab-separated columns with a header, `tip` then `date`
   (decimal year, e.g. `2011.5`). Write `NA` for samples with unknown
   collection date; those dates are then sampled within the observed range
   rather than dropped.

   ```
   tip	date
   A	2008
   B	2011.5
   C	NA
   ```

   Tip labels must match the tree exactly. Mismatched labels are silently
   unmatched, so check the `dates matched: n/total` line that is printed at
   startup.

## Run

```bash
# single chain: 10,000 iterations, thinning every 10 -> 1,000 stored samples
./target/release/rustbact tree.nwk dates.tsv 10000 10 --seed 42

# 8 parallel chains with checkpoints; writes run.bin.0 .. run.bin.7
./target/release/rustbact tree.nwk dates.tsv 10000 10 \
    --chains 8 --seed 200 --ckpt-out run.bin

# extend every chain by another 10,000 iterations, reusing all stored samples
./target/release/rustbact tree.nwk dates.tsv 20000 10 --chains 8 \
    --resume run.bin.0,run.bin.1,run.bin.2,run.bin.3,run.bin.4,run.bin.5,run.bin.6,run.bin.7
```

Positional arguments are `tree`, `dates`, then optionally `nbIts` (default
10000) and `thin` (default 10). Flags:

| Flag | Meaning |
|---|---|
| `--chains N` | run N independent chains in parallel threads (default 1) |
| `--seed N` | base seed; chain `i` uses `seed + i` |
| `--model NAME` | `arc` (default), `carc`, `poisson`, `negbin`, `strictgamma`, `relaxedgamma` |
| `--ckpt-out PATH` | write checkpoint(s); with `--chains N` writes `PATH.0` … `PATH.N-1` |
| `--resume PATH[,PATH…]` | resume chain(s) from checkpoint(s), one per chain |
| `--scale-lengths L` | multiply branch lengths by `L` before running; use the number of alignment sites to convert a per-site tree to substitutions |
| `--check-lengths` | print the total branch length, warn if it looks per-site, and exit |

On completion it prints a per-parameter summary (mean, sd, 95% HPD,
ESS per chain, total ESS, and split-R-hat across chains):

```
=== summary (8 chain(s), 8000 samples total) ===
param            mean           sd                 95% HPD   ESS/chain   ESS total    R-hat
mu             ...           ...          ...          ...          ...          ...
sigma          ...           ...          ...          ...          ...          ...
alpha          ...           ...          ...          ...          ...          ...
```

**How to read it:** R-hat below ~1.01 means the chains agree, so pooling them
is valid and the total ESS is meaningful. If R-hat is larger, the chains have
not converged — increase `nbIts`, or resume the checkpoints for more
iterations, until R-hat drops. Total ESS grows roughly linearly with the number
of chains, which is why several short parallel chains beat one long one.

## Performance

The per-iteration cost is dominated by how often the likelihood is
recomputed, which in BactDating grows with the node count (`O(n^2)` per
iteration). On a large dated phylogeny (tens of thousands of tips) this is the
difference between hours and minutes for a 10,000-iteration run.

| | BactDating (R) | rustbact |
|---|---|---|
| per-iteration likelihood work | full table | only branches touching the changed node |
| chains | one at a time | N in parallel threads, one core each |
| convergence diagnostics | needs a separate multi-chain setup | split-R-hat and ESS reported per run |

Concrete timings depend on the tree and the machine. On simulated trees the
per-iteration cost of the reference implementation fits roughly `t ∝ Nnode^1.7`:

| tips | nodes | s / iteration (BactDating) |
|---|---|---|
| 100 | 199 | 0.002 |
| 500 | 999 | 0.007 |
| 2000 | 3999 | 0.098 |
| 5000 | 9999 | 1.541 |

## Known differences from BactDating

Two issues in the R/C++ implementation surfaced while validating:

1. **`coalpriorC` reads out of bounds.** The loop needs `n` internal node
   dates but `bactdate()` passes only `n-1`, so the final iteration reads past
   the end of the vector (R warns `subscript out of bounds`). R's `alpha`
   estimate is therefore built on undefined behaviour. Passing an extra value
   makes R agree with this implementation exactly.
2. **`dnbinom(size = 0)`.** R returns a log-density of 0 at `x = 0` (a point
   mass) where a naive `lgamma`-based formula yields `NaN`. This shows up as a
   `NaN` likelihood whenever a branch has zero duration.

## Status / limitations

- The root position is fixed at the input tree's root (`updateRoot` moves are
  not implemented).
- Output is the parameter summary plus checkpoints; confidence intervals on
  node dates, DIC and the annotated tree are not yet exported.
- `loadGubbins()` / `loadCFML()` equivalents (recombination-aware inputs) are
  not implemented.

The distribution functions are unit-tested against values printed by R
(`dgamma`, `pgamma`, `dnbinom`, `dpois`, `lgamma`, and the coalescent prior), and
the incremental likelihood is tested to agree with a full recomputation to
1e-10.
