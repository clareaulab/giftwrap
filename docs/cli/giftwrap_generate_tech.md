---
name: The `giftwrap-generate-tech` Command
description: The `giftwrap-generate-tech` command is used to generate a scaffold for a technology-specific script that can be used to process GIFT-seq data. This command is useful for users who want to create custom processing scripts tailored to their specific sequencing technology.
---

# `giftwrap-generate-tech`
The `giftwrap-generate-tech` command is used to generate a scaffold for a technology-specific script that can be used to process GIFT-seq data. This command is useful for users who want to create custom processing scripts tailored to their specific sequencing technology.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap-generate-tech --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
