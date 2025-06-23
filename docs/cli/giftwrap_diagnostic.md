---
name: The `giftwrap-diagnostic` Command
description: The `giftwrap-diagnostic` command is used to do very basic mapping to get a quick estimate of GIFT-seq data quality. It is useful for troubleshooting mapping issues.
---

# `giftwrap-diagnostic`
The `giftwrap-diagnostic` command is used to do very basic mapping to get a quick estimate of GIFT-seq data quality. It is useful for troubleshooting mapping issues.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap-diagnostic --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
