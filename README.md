# Quarto Multi-Report Template

Self-contained Quarto website template. No external data files, no network
calls, no bibliography dependencies outside this folder.

## Run it

    quarto preview

Or open any `.qmd` in VS Code / RStudio and click **Preview**. Quarto detects
`_quarto.yml`, renders the whole site, and opens it with the sidebar working.

Output lands in `_site/`. Open `_site/index.html` to view offline.

## Files

| File | Purpose |
|---|---|
| `_quarto.yml` | Site config and sidebar. Add new reports here. |
| `_common.py` | Shared data generation, model, and map code. |
| `index.qmd` | Landing page. |
| `report-1.qmd` | Tables, summary stats, equations. |
| `report-2.qmd` | Model fitting, metrics, dual choropleth map. |
| `report-3.qmd` | Tabsets, callouts, figures, margin content. |
| `references/references.bib` | Placeholder citations. |
| `styles.css` | Optional cosmetic overrides. |

## Adding Report 4

1. Copy `report-3.qmd` to `report-4.qmd`.
2. Add to the `contents:` list in `_quarto.yml`:

       - text: "Report 4"
         href: report-4.qmd

## Using real data

Replace `make_records()` and `make_zones()` in `_common.py` with your own
loaders, keeping the returned column names the same. Everything downstream
continues to work.

## Dependencies

    pip install -r requirements.txt

Only `pandas`, `numpy`, `scikit-learn`, and `jupyter` are required. If
`xgboost` is missing the model falls back to scikit-learn; if `folium` is
missing the map falls back to matplotlib.

## Note

`self-contained: true` was removed from the original config because it is
incompatible with Quarto website projects. To get a single portable HTML file
instead of a site, drop the `project:`/`website:` blocks and render one
`.qmd` on its own.
