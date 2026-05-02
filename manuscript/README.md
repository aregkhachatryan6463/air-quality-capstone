# Manuscript (LaTeX)

## Build PDF

From this directory:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Figures: `main.tex` includes `../images/mae_by_horizon.png` (run `python run_pipeline.py` and `python export_paper_assets.py` from the project root so `images/` is populated).

## Sync before final submission

1. Run the pipeline (e.g. `python run_pipeline.py --skip-download --max-evals 10` from the repo root) and `python export_paper_assets.py`.
2. Copy split dates from `../results/json/split_info.json` if you edit ranges by hand, or re-run the pipeline and keep the automatic narrative in `main.tex`.
3. If `../images/mae_by_horizon.png` is missing, regenerate figures from `../results/tables/*.csv` or re-run the pipeline; `export_paper_assets.py` copies PNGs from `../results/plots/` when present.
