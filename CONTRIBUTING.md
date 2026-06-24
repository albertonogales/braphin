# Contributing to BRAPHIN

Thank you for your interest in contributing to BRAPHIN! This document explains how to report bugs, propose new features, and submit code changes.

---

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [Reporting bugs](#reporting-bugs)
3. [Requesting features](#requesting-features)
4. [Setting up the development environment](#setting-up-the-development-environment)
5. [Running the tests](#running-the-tests)
6. [Coding style](#coding-style)
7. [Submitting a pull request](#submitting-a-pull-request)
8. [Commit message conventions](#commit-message-conventions)

---

## Code of conduct

All contributors are expected to be respectful and constructive. Academic and scientific integrity matters — please do not submit plagiarised code or fabricated benchmark results.

---

## Reporting bugs

Before opening an issue, please:

1. Search existing issues to avoid duplicates.
2. Reproduce the bug against the latest `main` branch.
3. Include a **minimal reproducible example** (MRE) — synthetic or anonymised data only, never real patient data.

A good bug report contains:

- BRAPHIN version (`import braphin; print(braphin.__version__)`)
- Python version and OS
- Full traceback
- Minimal code snippet that reproduces the issue

---

## Requesting features

Open a GitHub issue with the `enhancement` label. Describe:

- The neuroimaging use case that motivates the feature
- Which pipeline stage it would affect
- Any relevant references (papers, toolbox implementations)

Features listed in the [Unreleased] section of `CHANGELOG.md` are already planned — contributions implementing them are especially welcome.

---

## Setting up the development environment

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/braphin.git
cd braphin

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"
# or equivalently:
pip install -r requirements-dev.txt && pip install -e .

# 4. (Optional) install EEG or GNN extras
pip install -e ".[eeg]"
pip install -e ".[gnn]"
```

---

## Running the tests

The test suite uses only synthetic data — no NIfTI files or internet access required.

```bash
# Run all 167 tests
pytest

# With coverage report
pytest --cov=braphin --cov-report=term-missing

# Run a specific test file
pytest tests/test_transform.py -v

# Run tests matching a keyword
pytest -k "centroid" -v
```

All tests must pass before a pull request will be reviewed. New code must be accompanied by new tests — aim for at least one happy-path test and one error-path test per public method.

---

## Coding style

- **PEP 8** with a line length of 100 characters.
- **Type hints** on all public function signatures (parameters and return type).
- **Docstrings** on all public classes and methods (Google-style preferred).
- **No silent stubs**: if a feature flag is not yet implemented, raise `NotImplementedError` with a clear message rather than silently returning the input unchanged.
- **Exception hierarchy**: raise the most specific subclass of `BRAPHINError` that fits the failure (see `braphin/exceptions.py`). Never raise a bare `Exception`.
- **Bundle pattern**: each pipeline stage must accept a typed bundle dataclass and return a new typed bundle dataclass. Do not mutate input bundles.
- **No real patient data** anywhere in the codebase — tests must use synthetic arrays generated with `numpy.random`.

---

## Submitting a pull request

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/bandpass-filtering
   ```
2. Make your changes, write tests, update `CHANGELOG.md` under `[Unreleased]`.
3. Ensure `pytest` passes with no failures or warnings.
4. Push and open a pull request against `main`.
5. Fill in the PR template: motivation, what changed, how it was tested.

Pull requests that break existing tests or reduce coverage without justification will not be merged.

---

## Commit message conventions

Follow the [Conventional Commits](https://www.conventionalcommits.org/) style:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

Types:

| Type | When to use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `test` | Adding or correcting tests |
| `docs` | Documentation only |
| `refactor` | Code change with no behaviour change |
| `chore` | Build scripts, CI, dependencies |

Examples:

```
feat(denoise): add temporal bandpass filtering via scipy.signal
fix(tabular): skip BIDS TSV header row to avoid NaN confound rows
test(transform): add centroid space assertions for voxel-only atlases
docs: add CONTRIBUTING guide
```

---

## Questions?

Open a GitHub discussion or contact the maintainer via the email listed in `pyproject.toml`.
