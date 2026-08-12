# Obtaining the Raw Data

The raw survey files (`w4_hh.csv`, `w4_adult.csv`) are **not included in this
repository**. They come from the World Bank Microdata Library under a
research-purposes-only licence that does not permit redistribution, so they must be
downloaded directly from the source rather than cloned from GitHub.

## Source
- **Dataset:** Kenya COVID-19 Rapid Response Phone Survey (RRPS)
- **Provider:** World Bank Microdata Library (in partnership with the Kenya National
  Bureau of Statistics)
- **URL:** https://microdata.worldbank.org
- **Access:** Free registration required; approved for research use.

## Steps
1. Create a free account on the World Bank Microdata Library.
2. Search for "Kenya COVID-19 Rapid Response Phone Survey."
3. Download the household-level and adult-level extracts for Wave 4.
4. Place the files in this repository at:
   ```
   data/raw/w4_hh.csv
   data/raw/w4_adult.csv
   ```
5. Run the notebooks in order starting from `notebooks/01_eda.ipynb`.

## Why this matters
This project only ever redistributes code and derived, aggregate results (model
weights, evaluation tables, figures) — never the underlying microdata itself, in
keeping with the licence terms. If you're extending this project, please confirm your
own use case is consistent with the Microdata Library's terms of use before
downloading.
