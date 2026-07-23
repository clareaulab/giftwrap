# Vendored report assets

These files are inlined into the step 5 HTML report (`giftwrap/html_report.py`) so
that a generated report renders with **no network access**. Reports are routinely
opened on offline laptops and cluster nodes, where loading these from a CDN would
silently produce a page with no figures.

| File | Version | Source | License |
|------|---------|--------|---------|
| `plotly.min.js` | 2.32.0 | https://cdn.plot.ly/plotly-2.32.0.min.js | MIT |
| `fonts.css` | — | Google Fonts: Figtree (300–700), JetBrains Mono (400, 500) | SIL Open Font License 1.1 |

`fonts.css` contains the **latin subset only**, with the `.woff2` files embedded as
`data:` URIs to keep the report a single portable file.

## Updating

To refresh plotly, download the desired version and update `_PLOTLY_VERSION` in
`html_report.py` (it is used only for the CDN fallback when these assets are absent):

```bash
curl -o src/giftwrap/assets/plotly.min.js https://cdn.plot.ly/plotly-<version>.min.js
```
