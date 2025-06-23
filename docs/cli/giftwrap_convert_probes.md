---
name: The `giftwrap-convert-probes` Command
description: The `giftwrap-convert-probes` command is used to convert a 10X Cell Ranger probe set to a GIFTwrap-compatible probe set file. This is useful for integrating existing 10X data into the GIFT-seq workflow.
---

# `giftwrap-convert-probes`
The `giftwrap-convert-probes` command is used to convert a 10X Cell Ranger probe set to a GIFTwrap-compatible probe set file. This is useful for integrating existing 10X data into the GIFT-seq workflow.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap-convert-probes --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
