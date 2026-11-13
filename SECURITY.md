# Security

## Reporting a vulnerability

Please do not open a public issue. Email **lucasargento7@gmail.com** (subject: "zerograd security") with what you found, how to reproduce it and
what it affects, or use *Report a vulnerability* in the repository's Security tab. We answer within 2 business days and keep you posted until it is fixed. If you want credit in
the release notes, say so.

## What the SDK does with your data

* **Local mode** (`ZeroGrad()` without an API key): your data never leaves your machine. The SDK only makes
  network requests to download model weights from each author's official Hugging Face repository.
* **Remote mode** (`remote=True`): your data is sent over HTTPS to the ZeroGrad platform (`api.zerograd.dev`).
  `zerograd.privacy` (`scan`, `drop_sensitive`, `Vault`) runs on your machine before anything is sent.
* The SDK collects no telemetry.
* A capability file written by `cap.save()` contains a sample of your labeled rows (the model's context).
  Treat it like the data itself.

## Supply chain

Releases are built and published to PyPI by GitHub Actions with trusted publishing (no long-lived tokens), from a
version tag on the default branch. PyPI attaches build attestations to every file.

## Supported versions

Security fixes go to the latest minor release.
