# Manuscript outputs

- `tables/` — LaTeX tables from `python export_paper_assets.py` (use `\usepackage{booktabs}` in the preamble)
- `results_summary.json` — compact summary for manuscript writing
- `bundle/` — Test-set predictions (local only; not committed to git). If present, it is used only as a **fallback** for the bootstrap table when `results/json/results_summary.json` does not contain a `bootstrap_mae_95ci` object.

- Pipeline **figures**: `results/plots/` (from `python run_pipeline.py`), mirrored to **`images/`** for thesis embeds. `export_paper_assets.py` also copies the same key PNGs into `images/`.
- Pipeline **numeric tables (CSV)**: `results/tables/` — `export_paper_assets.py` reads these and writes LaTeX into `paper/tables/`.
- **Bootstrap** rows for `table_bootstrap_mae_ci.tex` are taken from **`results/json/results_summary.json`** when that file contains a dict `bootstrap_mae_95ci` (preferred over the legacy bundle).
See `docs/METHODOLOGY.md` and `results/README.md` for layout.
