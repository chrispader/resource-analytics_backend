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
python -m experiments.role_matrix.generate --output generated/role-matrix
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
