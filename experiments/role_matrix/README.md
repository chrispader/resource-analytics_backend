# Resource × Role matrix experiment datasets

This package creates the synthetic event logs used to compare matrix
orderings. The design varies four factors independently:

- resource count: 40 (small), 100 (medium), and 400 (large);
- role count: 10, 20, and 40;
- target density: 0.15, 0.35, and 0.60;
- profile diversity: low, medium, and high.

Five deterministic seeds are generated for every combination. The complete
factorial design therefore contains 405 logs (3 × 3 × 3 × 3 × 5). Keeping the
three resource counts as size groups makes it possible to study size without
confounding it with role count, density, or profile diversity.

For the three diversity levels of one otherwise identical condition, row
degrees and the total number of filled cells stay unchanged. Diversity changes
only how many distinct role profiles occur within five balanced, planted
organizational groups and how evenly resources are spread over those profiles.
Group membership is identical across the three diversity levels. Profiles favor
the subset of roles assigned to their latent group, while still allowing role
overlap between groups. Every resource has at least one role and every
configured role is represented. `Profile Group` records the latent group rather
than the resource's exact observed role profile.

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

The results include fixed-ordering metrics, random-baseline metrics, dataset
and color summaries, all rule-level heuristic outcomes, direction-aware random
comparisons, marginal factor-level aggregates, across-seed condition
aggregates, and JSON metadata/configuration. Before evaluation, each CSV's
SHA-256 hash, observed resource and role counts, and density are checked against
the manifest. This prevents a changed or incomplete input from silently entering
the results. The random comparison treats lower fragmentation as better and all
other metrics as higher-is-better. `--random-seeds` can reduce the baseline
count for a quick development run; the research default is 100.
