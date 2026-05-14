# Manuscript (LaTeX)

## ICDSG 2026 submission notes

Target venue: [ICDSG 2026 — 2nd International Conference on Data Science and Geoinformatics](https://www.icdsg.org/2026/).

- **Length:** 4--6 pages **including** figures, tables, and references (IEEE double-column).
- **Template:** Official IEEE conference LaTeX (conference mode) from [IEEE conference templates](https://www.ieee.org/conferences/publishing/templates.html); this repo uses vendored `IEEEtran.cls` / `IEEEtran.bst` compatible with that guidance.
- **Review:** **Double-blind** --- no author names or affiliations in the PDF for review. In `main.tex`, `\icdsgreviewtrue` enables the anonymous author block and strips acknowledgments; set `\icdsgreviewfalse` (or comment the line) for camera-ready.
- **Submit:** EDAS --- [new paper link for conference 34472](https://edas.info/newPaper.php?c=34472) (per ICDSG site).
- **Originality:** iThenticate similarity above 20% may be rejected per venue policy; plan a pre-check.
- **Camera-ready:** IEEE PDF eXpress (conference ID announced on the venue page) before final upload.

## IEEE conference paper (default `main.tex`)

The submission target is **IEEEtran** (`\documentclass[conference]{IEEEtran}`), **two-column**, **six pages** including references in the current build (ICDSG allows 4--6 pages total—stay within the CFP cap).

**Floats:** `dblfloatfix.sty` is vendored here (TinyTeX may not ship it) so wide `table*` / `figure*` can use bottom placement and stay in order. A `\clearpage` before **Discussion** flushes Result floats so conclusions do not precede the last figures.

**Build** (from this directory):

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

**Templates:** `IEEEtran.cls` and `IEEEtran.bst` are vendored in this folder (LPPL, Michael Shell) so the project compiles even when a TeX distribution omits the `ieeetran` package. You may replace them with your TeX tree’s copies if you prefer.

**Figures:** `main.tex` expects PNGs under `../images/` (e.g. `stations_map_lonlat.png`, `acf_pacf_train.png`, `mae_by_horizon.png`, `spatial_level_comparison_mae.png`, `walk_forward_mae_stability.png`). Populate `images/` via `run_pipeline.py` / `export_paper_assets.py`, or copy from a `replication_bundle/*/images/` snapshot.

**Tables:** Numeric fragments for the short paper live in `../paper/tables/ieee/*.tex`. After a new pipeline run, copy updated numbers from `../paper/tables/table_main_results.tex` (and related CSVs) into those fragments, or extend `export_paper_assets.py` to regenerate them automatically.

**Camera-ready notes:** IEEEtran reminds you to balance column lengths on the last page and use Type~1 fonts only. Some conferences ship a **vendor-specific** LaTeX kit (margin tweaks, copyright line)—swap in that kit when the CFP requires it.

## Long single-column draft

The previous full article (single-column `article` class) is preserved as **`main_long.tex`** for appendices and extended narrative.

## TinyTeX / missing fonts

If `\texttt{...}` or Courier fails to build, prefer plain text or `\emph{...}` for file names (the IEEE short paper avoids typewriter for this reason). For `graphicx` issues, see TeX Live’s `grfext` package.
