# Manuscript outputs

- `tables/` — LaTeX tables from `python export_paper_assets.py` (use `\usepackage{booktabs}` in the preamble)
- `results_summary.json` — compact summary for manuscript writing
- `bundle/` — Test-set predictions (local only; not committed to git)

Primary model figures are generated in `results/plots/` from `python run_pipeline.py`.
See `docs/METHODOLOGY.md` for a concise description of the active pipeline.
