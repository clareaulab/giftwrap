---
name: The `giftwrap-generate-r` Command
description: The `giftwrap-generate-r` command is used to generate a scaffold for an R script that can load processed GIFT-seq data into a Seurat object.
---

# `giftwrap-generate-r`
The `giftwrap-generate-r` command is used to generate a scaffold for an R script that can load processed GIFT-seq data into a Seurat object. This command is useful for users who want to analyze their GIFT-seq data using the Seurat package in R.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap-generate-r --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
