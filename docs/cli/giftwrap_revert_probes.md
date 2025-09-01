---
name: The `giftwrap-revert-probes` Command
description: The `giftwrap-revert-probes` command is used to convert a GIFTwrap-compatible probe set file to a 10X Cell Ranger probe set. This can be useful for comparing to ranger-processed data.
---

# `giftwrap-revert-probes`
The `giftwrap-revert-probes` command is used to convert a GIFTwrap-compatible probe set file to a 10X Cell Ranger probe set. This can be useful for comparing to ranger-processed data.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap-revert-probes --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
