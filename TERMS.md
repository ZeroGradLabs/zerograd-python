# Terms of use for models

The ZeroGrad SDK is open source under the Apache License 2.0. The **models** it can run are not part of
this package and are not licensed by ZeroGrad. Each one comes with its author's license, and that license
is what governs how you may use it.

## Your responsibility

* **You are responsible for complying with the license of every model you use**, including attribution
  requirements and any restriction on commercial use. `zerograd.list_models()` shows each license, a link
  to it, and whether it allows commercial use; read the license itself before relying on it.
* ZeroGrad does not host, mirror or redistribute model weights. The SDK downloads them from each author's
  official repository, to your machine, with your own account and credentials where the author requires them.
* ZeroGrad's summary of a license (for example "commercial use: yes") is guidance, not legal advice, and can
  be out of date. The author's license text prevails.

## Non-commercial models

Some models (for example TabPFN 2.5 and later, TabFM, EXAONE Tabular and LimiX-2) are released under licenses
that **forbid commercial use**. In this SDK they:

* are **off by default**, and run only when you name each one: `ZeroGrad(noncommercial=["tabpfn-2.5"])`;
* show a warning with their license every time they are enabled;
* are downloaded with **your own account**, accepting the author's license where the author asks for it
  (TabPFN 2.5+ asks you to sign in and accept Prior Labs' license in the browser);
* **never run on the ZeroGrad platform**: `remote=True` refuses them;
* are meant for **research and evaluation only**. Do not use them in production, in a product, or in any
  service you charge for, unless you have a commercial license from their author.

Capabilities saved with a non-commercial champion refuse to open unless you name that model again.

## No warranty

The SDK and its results are provided "as is", without warranty of any kind, as stated in the Apache License 2.0.
Predictions can be wrong; evaluate them on your own data (the championship's holdout is a start, not a guarantee).

## Your data

In local mode (the default without an API key) your data never leaves your machine. With `remote=True` your data
is sent to the ZeroGrad platform and is governed by the platform's terms and privacy policy. The privacy tools in
`zerograd.privacy` (`scan`, `drop_sensitive`, `Vault`) run on your machine before anything is sent.
