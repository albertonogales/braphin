## Summary

<!-- What does this PR do? One or two sentences. -->

## Motivation

<!-- Why is this change needed? Link to the related issue if one exists: Closes #NNN -->

## Changes

<!-- Bullet list of what changed. Be specific about which files and classes were modified. -->

- 
- 

## Pipeline stage affected

- [ ] Stage 1 — InputMRIData
- [ ] Stage 2 — PreprocessMRIData
- [ ] Stage 3 — DenoiseMRIData
- [ ] Stage 4 — TransformMRIData
- [ ] Stage 5 — ModelMRIConnectivityData
- [ ] Atlas / helpers
- [ ] Configuration dataclasses
- [ ] Exceptions
- [ ] Unified API (eegraph)
- [ ] Documentation
- [ ] CI / packaging

## Testing

<!-- How was this tested? All new code must have tests. -->

- [ ] New unit tests added in `tests/`
- [ ] Existing 167 tests still pass (`pytest`)
- [ ] Tests use synthetic data only (no real NIfTI files)
- [ ] Coverage did not decrease

## Checklist

- [ ] Type hints on all new public function signatures
- [ ] Docstrings on all new public classes and methods
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No real patient data committed
- [ ] `ruff check braphin/` passes with no errors
