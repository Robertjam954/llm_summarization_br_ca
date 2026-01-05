# Data Project Template

<a target="_blank" href="https://datalumina.com/">
    <img src="https://img.shields.io/badge/Datalumina-Project%20Template-2856f7" alt="Datalumina Project" />
</a>

## Cookiecutter Data Science
This project template is a simplified version of the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org) template, created to suit the needs of Datalumina and made available as a GitHub template.

## Adjusting .gitignore

Ensure you adjust the `.gitignore` file according to your project needs. For example, since this is a template, the `/data/` folder is commented out and data will not be exlucded from source control:

```plaintext
# exclude data from source control by default
# /data/
```

Typically, you want to exclude this folder if it contains either sensitive data that you do not want to add to version control or large files.

## Duplicating the .env File
To set up your environment variables, you need to duplicate the `.env.example` file and rename it to `.env`. You can do this manually or using the following terminal command:

```bash
cp .env.example .env # Linux, macOS, Git Bash, WSL
copy .env.example .env # Windows Command Prompt
```

This command creates a copy of `.env.example` and names it `.env`, allowing you to configure your environment variables specific to your setup.


## Project Organization

```
├── LICENSE                        <- Open-source license if one is chosen
├── README.md                      <- The top-level README for developers using this project
├── data
│   ├── external                   <- Data from third party sources
│   ├── interim                    <- Intermediate data that has been transformed
│   ├── processed                  <- The final, canonical data sets for modeling
│   └── raw                        <- The original, immutable data dump
│
├── notebooks/                     <- Jupyter notebooks for exploration and prototyping
├── references/                    <- Data dictionaries, manuals, and other explanatory materials
├── src/                           <- Source code for this project
    │
    ├── data collection and processing/   <- Data collection, cleaning, and processing code
    │   ├── collection                      <- Gather data from databases, APIs, web scraping, and other sources
    │   │   (Collection: gather data from various sources like databases, APIs, or web scraping.)
    │   └── cleaning                        <- Cleaning & processing routines
    │       (Cleaning & Processing: encode variables, handle missing values, remove duplicates, transform data, and prepare it for analysis.)
    │
    ├── data reports/                <- Generated analysis outputs and exportable tables (reports: HTML/PDF/LaTeX/etc.; tables: CSV/TSV/Excel)
    │   ├── reports                  <- Rendered analysis outputs (HTML, PDF, LaTeX, etc.)
    │   └── tables                   <- CSV/TSV/Excel tables and summary tables used for manuscripts/reports
    │
    ├── exploratory data analysis/   <- EDA: use statistics and visualizations to understand data patterns and relationships
    │   (Exploratory Data Analysis (EDA): use statistics and visualizations to understand data patterns and relationships.)
    │
    ├── modeling/                    <- Choose, train, and evaluate both associational (statistical) and predictive (machine learning) models
    │   ├── train.py                  <- Code/scripts to train models
    │   ├── predict.py                <- Code to run model inference with trained models
    │   └── evaluation.md             <- Notes or scripts for evaluating model performance and reporting results
    │   (Modeling: Includes associational/statistical analyses to estimate relationships (effect estimates, confidence intervals, model diagnostics) AND predictive/ML models to forecast or classify. Evaluate predictive models using metrics like precision, recall, and F1-score; report associational results using appropriate statistical summaries and diagnostics.)
    │
    ...existing code...
    └── services/                    <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py
└── manuscript components/         <- Manuscript drafts, figures, and text components for papers
```
