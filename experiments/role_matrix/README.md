# Resource × Role matrix experiment datasets

This package creates the synthetic event logs used to compare matrix
orderings. The design varies four factors independently:

- resource count: 40 (small), 100 (medium), and 400 (large);
- role count: 10, 20, and 40;
- target density: 0.15, 0.35, and 0.60;
- profile diversity: low, medium, and high.

Five deterministic seeds are treated as blocked replicates for every
combination. The complete
factorial design therefore contains 405 logs (3 × 3 × 3 × 3 × 5). Keeping the
three resource counts as size groups makes it possible to study size without
confounding it with role count, density, or profile diversity.

The role levels deliberately start at 10 rather than 5. At density 0.15, a
five-role matrix contains only 0.75 assignments per resource on average. That
is mathematically incompatible with the requirement that every represented
resource has at least one role. Ten roles allow all resources to remain in the
matrix while preserving the requested low-density condition.

For the three diversity levels of one otherwise identical condition, the exact
row-degree sequence and total number of filled cells stay unchanged. Degree
generation starts from low, central, and high target bands weighted 25/50/25.
Boundary clipping and integer correction can merge or change those bands, so
the realized proportions are not claimed to remain 25/50/25. The manifest
records the exact distribution summaries, and the generator guarantees only
that the final distribution is non-degenerate and has the requested assignment
total.
Resource-to-group assignment, degree placement within groups, and profile
placement use separately salted deterministic random streams. This prevents the
alphabetical resource identifier from carrying the planted degree or profile
order. Each exact degree value is also stratified across the three planted
groups, so its group counts differ by at most one. The manifest records
Cramér's V for the realized degree-by-group table. The exhaustive design test
requires mean alphabetical degree-order agreement across all 405 logs to stay
between 0.45 and 0.55, checks every full factor cell across its five seed
blocks, bounds degree/group association and alphabetical group contiguity, and
compares alphabetical profile coherence with an independently shuffled order.

Diversity is controlled along the deterministic generator's tested
profile-count path for each fixed condition. The first generated assignment on
that path is the baseline and the last is the maximum generated endpoint; these
are not claimed to be mathematical extrema over all possible matrices. Low,
medium, and high target relative positions of 0.25, 0.60, and 0.90 between the
two generated entropy endpoints, with a ±0.12 tolerance for discrete small
conditions. Raw normalized profile entropy and both endpoints are exported so
the relative value remains auditable across resource counts.

Three balanced latent organizational groups are planted independently of
diversity. Each group always includes its own exclusive signature role and
excludes the other groups' two signature roles. Remaining assignments come
from a shared role pool with a mild, deterministic group preference. The
generator requires mean within-group Jaccard similarity to exceed mean
between-group similarity by at least 0.05. `Profile Group` records the latent
group for experiment-only validation; it is never passed to an ordering
algorithm.

## Generate the logs

Run this command from the backend repository root:

```sh
python -m experiments.role_matrix.workflow generate \
  --output generated/role-matrix
```

The command writes logs into `small/`, `medium/`, and `large/` directories and
creates `manifest.csv`. The manifest records all experimental factors, realized
density, unique-profile count, normalized profile entropy, and a SHA-256 hash
for every CSV, together with the number of planted groups. The command fails if
diversity is not strictly increasing from
low to medium to high for any matched condition.

The committed `config.json` is the experiment definition. Pass a smaller test
configuration with `--config` when developing the workflow; do not manually
edit generated CSV files.

## Evaluate the logs

The reusable runner uses the same matrix construction, four fixed orderings,
structural metrics, color evaluation, Plotly renderer, and 23-rule heuristic
catalog as the application. It also evaluates 100 deterministic random
permutations per dataset and computes an experiment-only planted-group
contiguity score. Run it without a server or browser:

```sh
python -m experiments.role_matrix.workflow evaluate \
  --manifest generated/role-matrix/manifest.csv \
  --output generated/role-matrix-results
```

Use the `full` command to generate and evaluate in one reproducible operation:

```sh
python -m experiments.role_matrix.workflow full \
  --output generated/role-matrix-experiment
```

The results include fixed-ordering metrics, descriptive random-baseline
metrics, dataset
and color summaries, all rule-level heuristic outcomes, direction-aware random
comparisons, marginal factor-level aggregates, across-seed condition
aggregates, pairwise factor-interaction summaries, and JSON
metadata/configuration. The interaction table is intended for descriptive
analysis of the factorial design; it does not by itself support inferential
claims. Before evaluation, the manifest is checked against the full Cartesian
product in the configuration, and each CSV's
SHA-256 hash, observed resource and role counts, and density are checked against
the manifest. The evaluator also recomputes unique profiles, profile entropy,
group count and separation, and the degree-distribution summaries from each
matrix. It rejects differences and verifies the configured diversity tolerance
and minimum group separation. This prevents changed or internally inconsistent
input from silently entering the results. The random comparison treats lower fragmentation as better and all
other metrics as higher-is-better. `--random-seeds` can reduce the baseline
count for a quick development run; the research default is 100.

Run metadata records input and output hashes, relevant source-file hashes, the
backend Git commit and dirty state when available, the Python version, and key
dependency versions. These fields connect a result directory to the precise
data, configuration, and implementation used to create it. Source provenance
covers the directly invoked local modules and dependency declaration; it is not
presented as a hash of the complete operating-system environment.
Unknown files already present in an output directory are left untouched and
listed as ignored in the run metadata; the workflow only overwrites its named
artifacts. Known artifacts are first completed in a temporary sibling staging
directory and are published only after every computation and metadata write
succeeds, so a failed run does not overwrite the previous valid results.
