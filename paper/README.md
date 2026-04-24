# Manuscript outputs

- `tables/` — LaTeX tables from `python export_paper_assets.py` (use `\usepackage{booktabs}` in the preamble)
- `results_summary.json` — compact summary for manuscript writing
- `bundle/` — Test-set predictions (local only; not committed to git)

- Pipeline **figures**: `results/plots/` (from `python run_pipeline.py`).
- Pipeline **numeric tables (CSV)**: `results/tables/` — `export_paper_assets.py` reads these and writes LaTeX into `paper/tables/`.
See `docs/METHODOLOGY.md` and `results/README.md` for layout.
