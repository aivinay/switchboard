# Releasing Switchboard

How to cut a versioned release of `switchboard-local` to PyPI.

## Pre-release checklist

- [ ] `make check` is green (ruff + mypy + full test suite).
- [ ] Version bumped in `pyproject.toml` (`[project] version`) and a matching
      `CHANGELOG.md` entry moved out of `[Unreleased]`.
- [ ] `README.md` "Proof" numbers reflect the latest benchmark aggregate.

## 1. Verify exactly what will ship

Run this from the repo root and eyeball the list — it must contain **no**
private data (`switchboard.db`, `reports/`, `docs/publication_plan.md`,
`docs/research_plan.md`, `docs/reviews/`, the whole `docs/paper/` folder,
`.env`, `data/*.jsonl`):

```bash
git ls-files | sort
git ls-files | grep -iE 'switchboard\.db|^reports/|publication_plan|research_plan|reviews/|^docs/paper/|\.env$|/data/.*\.jsonl' \
  && echo "STOP: private file is tracked — fix .gitignore" || echo "clean"
```

## 2. Build + check locally

```bash
.venv/bin/python -m pip install --upgrade build twine
.venv/bin/python -m build           # writes sdist + wheel to dist/
.venv/bin/python -m twine check dist/*
```

## 3. Tag and release

```bash
git tag -a v0.1.0 -m "Switchboard 0.1.0"
git push origin v0.1.0
```

Then create a **GitHub Release** for that tag. Publishing the release triggers
`.github/workflows/release.yml`, which builds and uploads to PyPI.

## First-time PyPI setup (once)

1. Confirm the distribution name **`switchboard-local`** is available
   (https://pypi.org/project/switchboard-local/ -> 404 means free). If taken,
   pick another name and update `pyproject.toml` `[project] name` + the
   README/CI badges.
2. On PyPI, add a **Trusted Publisher** for the project pointing at
   `aivinay/switchboard` and workflow `release.yml`.
3. In the GitHub repo, create an environment named **`pypi`**.

No API token is needed — publishing uses OIDC trusted publishing.

## Post-release

```bash
pip install switchboard-local         # from a clean venv, confirm it installs
switchboard doctor
```

- Verify the release on https://pypi.org/project/switchboard-local/.
- Start a new `[Unreleased]` section in `CHANGELOG.md`.
