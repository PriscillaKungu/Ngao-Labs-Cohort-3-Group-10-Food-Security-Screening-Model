# Predicting Household Food Insecurity in Kenya

## Overview
This project classifies Kenyan households into four categories: food secure, mildly insecure, moderately insecure, and severely insecure. It uses data from a nationally representative COVID-19-era phone survey. The goal is a low-cost screening model that helps humanitarian organizations prioritize outreach when resources are limited such as ahead of a forecast climate shock. An XGBoost classifier with class-balanced weights was selected after comparison against a majority-class baseline, a hyperparameter-tuned variant, and a CatBoost model. It achieved a macro F1 of 0.454 against a baseline of 0.167.

## Problem Statement
Predicting a Kenyan household's food status (food secure/ mildly insecure/moderately insecure/severely insecure) at the time of Wave 4 of the COVID-19 Rapid Response Phone Survey, using household demographics, housing characteristics, asset ownership, employment, and coping-behavior data collected in the same survey wave. Primary metric: Macro F1, chosen to weight all four classes, including the small, high-priority severely-insecure class, equally rather than letting the majority food-secure class dominate the score. 


## Dataset
Name: Kenya Covid-19 Rapid Response Phone Survey (RRPS), Wave 4
Source: World Bank Microdata Library
Size: 4,894 households (w4_hh.csv) + 10,199 linked adult records (w4_adult.csv)
Target: food_security_status- a 4-class label derived from a 7-question Food Insecurity Severity Score (FISS) covering worry, hunger, skipped meals, and going a full day without food for adults and children
Class balance: Food secure: 50%, Mildly insecure: 24.3%, Moderately insecure: 13.2%, Severely insecure: 12.6%

License: Research use only; redistribution not permitted. Raw files are not included in this repository- see data/raw/data_download.md for how to obtain them directly from source. 

## Methods
1. EDA: notebooks/01_eda.ipynb
   Loaded and merged household and adult-level tables, constructed the FISS-    based target, and explored its relationship to demographics, housing, and     geography
2. Preprocess: notebooks/02_preprocessing.ipynb
     Engineered features covering assets, farming/livestock, enterprise activity, transfers, expenditure shares, and dependency ratio
   Screened candidate features for excessive missingness
   one-hot encoded categoricals
   Performed a stratified train/test split
   Applied leakage-safe county level target encoding (fit on the training split only)
3. Modelling: notebooks/03_modelling.ipynb
   Compared a majority-class baseline against a default XGBoost classifier with class-balanced sample weights, a RandomizedSearchCV-tuned XGBoost, and a CatBoost model using native categorical handling.
4. Evaluation: notebooks/04_evaluation.ipynb
   Macro and weighted F1
   A disaggregated evaluation by urban/rural status and county
   Structured error analysis of the confusion matrix

Results

| Metric | Majority-class baseline | Final model (XGBoost, default + balanced weights) |
| --- | --- | --- |
| Macro F1 | 0.167 | 0.454 |
| Weighted F1 | 0.333 | 0.538 |

A RandomizedSearchCV-tuned XGBoost (macro F1 0.451) and a CatBoost model with native categorical handling (macro F1 0.446) were also evaluated; neither improved on the default configuration, which was retained as the final model.

Disaggregated Performance

|Subgroup| N | Macro F1| Weighted F1| 
|---| ---| ---|---|
|urban| 523| 0.469| 0.563|
|Rural| 456| 0.436| 0.507|

Performance also varies by county; several counties with adequate test-set size N>= 20) fall below the overall average (e.g. macro F1 as low as 0.247 - 0.286) while others exceed it. See notebooks/04_evaluation.ipynb for the full table.

## Limitations
- Trained on 2020-2021 pandemic-era data; not validated against a climate or economic shock of a different kind, and predictions should be re-calibrated against more recent data before any real-world use
- Weakest on moderately-insecure class (precision 0.24), which sits in the ambiguous middle of an ordinal severity scale
- Rural households have both lower overall macro F1 (0.436 vs 0.469 urban) and lower recall specifically on the severely-insecure class (0.459 vs 0.581)- see the Responsible AI Statement for what this means for deployment.
- The model uses standard multi-class classification, which does not exploit the natural ordering of the four severity classes; an ordinal model is unexplored future work.

## How to Run
install -r requirements.txt

Place w4_hh.csv and w4_adult.csv in data/raw/ (see data/raw/data_download.md), then run the notebooks in order:
jupyter notebook notebooks/01_eda.ipynb   # then 02, 03, 04 in sequence

Each notebook saves its output to data/processed/ for the next notebook to load.



