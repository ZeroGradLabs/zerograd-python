# Releasing

Step by step for the first public release and every one after it. Commands are for macOS or Linux.

## 0. Once: accounts

1. **GitHub**: the org `zerogradlabs` exists and has an empty public repo `zerograd-python` (no README, no license,
   no .gitignore: the code brings its own).
2. **PyPI**: log in at pypi.org → *Your account* → *Publishing* → *Add a new pending publisher*:
   * PyPI project name: `zerograd`
   * Owner: `zerogradlabs` · Repository: `zerograd-python`
   * Workflow name: `release.yml` · Environment name: `pypi`
3. **GitHub environment**: in the repo, *Settings* → *Environments* → *New environment* → `pypi`
   (optionally add yourself as a required reviewer so every release waits for your click).

## 0b. When the repo goes public

Both settings are unavailable while the repo is private on the free org plan:

1. *Settings* → *Advanced Security* → turn on **Private vulnerability reporting** (SECURITY.md points to it).
2. *Settings* → *Environments* → `pypi` → add `lucasargento` as a **required reviewer**
   (the environment already exists, limited to `v*` tags).

## 1. Once: push the code

The repo's history comes from `zerograd-python.bundle` (in the project files, `oss/`).

```bash
git clone zerograd-python.bundle zerograd-python
cd zerograd-python
git remote set-url origin git@github.com:zerogradlabs/zerograd-python.git
git push -u origin main
```

CI runs on the push (tests on Python 3.10 to 3.13, and a build check). Wait for it to go green.

## 2. Try it on a clean machine

```bash
python3 -m venv /tmp/zg && source /tmp/zg/bin/activate
pip install ./                                  # from the repo checkout
python examples/demo.py                         # downloads the weights the first time
```

And open the notebook in Colab with the README's *Open in Colab* button (the repo must be public for it to work).

## 3. Benchmark (optional, before the big launch)

```bash
pip install -e ".[all]"
python benchmarks/run.py                        # ~30-60 min on a laptop CPU; writes benchmarks/results.csv
```

Paste the printed table into the launch posts. Keep the losses in.

## 4. Re-record the GIF with every model

```bash
brew install vhs                                # https://github.com/charmbracelet/vhs
pip install -e ".[all]"
python examples/demo.py                         # once, so the weights are cached and the GIF has no downloads
vhs scripts/demo.tape                           # writes docs/demo.gif
git add docs/demo.gif && git commit -m "Demo GIF with every model" && git push
```

## 5. Publish a version

1. Set the version in `zerograd/_version.py` and add its entry to `CHANGELOG.md`. Commit and push to `main`.
2. Tag it (the tag must match the version, or the release stops):

```bash
git tag v0.2.0
git push origin v0.2.0
```

The `Release` workflow runs the tests, builds, publishes to PyPI with trusted publishing (no tokens) and creates
the GitHub release with the files. A minute later `pip install zerograd` installs it.

If something is wrong after publishing: PyPI does not let you reuse a version. Fix it, bump to `0.2.1`, tag again.
You can *yank* the broken version on pypi.org so new installs skip it.
