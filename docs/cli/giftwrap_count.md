---
name: The `giftwrap-count` Command
description: The `giftwrap-count` command is used to map GIFT-seq reads to the probe set into a flat file.
---

# `giftwrap-count`
The `giftwrap-count` command is used to map GIFT-seq reads to the probe set into a flat file. This command is typically run after the initial sequencing data has been generated and is essential for quantifying the expression of genes or other targets in the GIFT-seq workflow.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap-count --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
