# Contributing

Thanks for helping. This repo uses **GitHub flow**: `main` is always releasable, every change goes through a
short-lived branch and a pull request, and releases are tags on `main`. There is no `dev` branch.

## Branches

Branch off the latest `main` and name the branch after the kind of change:

| Prefix | For | Example |
| --- | --- | --- |
| `feature/` | new functionality | `feature/uplift` |
| `fix/` | a bug fix | `fix/tabpfn-cpu-limit` |
| `hotfix/` | an urgent fix for a published release | `hotfix/0.2.3-import-error` |
| `docs/` | docs, README, notebooks | `docs/use-cases-notebook` |
| `chore/` | CI, tooling, dependencies | `chore/ruff-0.7` |

Branches named `claude/*` are opened by Claude Code sessions and count as `feature/`.
`artifacts/<run id>` branches are written by the *Collect artifacts* workflow and are temporary.

## Pull requests

1. Open a PR against `main`. Keep it to one change; small PRs get merged faster.
2. CI must be green: `ruff check .` and `pytest -q` on Python 3.10 to 3.13, plus the package build.
   Run them locally first: `uv pip install -e ".[dev]" && ruff check . && pytest -q`.
3. Add a line to `CHANGELOG.md` under *Unreleased* when users will notice the change.
4. PRs are **squash merged**, so the PR title becomes the commit on `main`: write it as a short sentence.
5. The branch is deleted when the PR merges. Need more changes later? Branch again from `main`.

## Releases and hotfixes

Releases are tags: bump the version, merge to `main`, then push a `v*` tag; `release.yml` publishes to PyPI
after a maintainer approves the `pypi` environment. Step by step in [RELEASING.md](RELEASING.md).

A hotfix is the same flow, only faster: `hotfix/...` from `main`, PR, merge, bump the patch version, tag.
We don't patch old releases on side branches; the fix ships in the next patch version.

## Branch cleanup

The weekly *Branch cleanup* workflow deletes branches that are merged, branches whose PR was closed without
merging and that have been idle for 14 days, and `artifacts/*` older than 30 days. It never touches `main`,
protected branches or branches with an open PR. Other idle branches are only listed in the run summary.
A deleted PR branch can be restored from its PR page (*Restore branch*).

## Security

Please report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md), not in a public issue.
