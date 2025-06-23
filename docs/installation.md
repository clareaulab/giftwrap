---
title: Installation Guide
summary: Instructions for installing GIFTwrap.
---
# Installation
GIFTwrap is published on [PyPI](https://pypi.org/project/giftwrap/) and can be installed using `pip`. It is compatible with Python 3.11 and later versions.

!!! warning
    Ensure that you install `giftwrap-sc`, NOT `giftwrap`, which is an unrelated package.

Since the package provides a CLI, it is recommended to install it with `pipx` or `uvx` if you do not plan to use the
python API, so that the CLI is available on your PATH and does not interfere with your current environment. In this case
you can replace `pip` with `pipx` or `uvx` in the following commands.

Recommended installation:
```bash
pip install giftwrap-sc[all]
```
This installs the requirements for spatial data processing and basic summary analyses of paired WTA data.

Alternatively, you can install with fewer dependencies as follows:
```bash
pip install giftwrap-sc[analysis]  # For additional basic summary analysis
pip install giftwrap-sc[spatial]  # For spatial counts processing
pip install giftwrap-sc  # Minimal installation
```

## Note
It is recommended that you have either `cellranger` or `spaceranger` installed in your system PATH if you plan to use the CLI as `giftwrap` will automatically detect cell barcode whitelists from these tools. If unavailable, it will try to revert to the `cellranger` default whitelist.
