# Rossmann Retail Sales Forecasting

This project forecasts daily sales at the store level for retail planning.
It combines historical sales and store data in SQLite and uses opening
schedules, planned promotions, and calendar information to predict sales.

## Data

Download `train.csv` and `store.csv` from the [Rossmann Store Sales competition on Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data). Extract the files and place them at these paths relative to the project root.

| File | Location | Contents |
| --- | --- | --- |
| `train.csv` | `data/raw/train.csv` | Daily store sales, opening status, promotions, and holidays from January 2013 through July 2015 |
| `store.csv` | `data/raw/store.csv` | Store metadata, including store type, assortment, and competitor information |

The notebook joins both files on `Store` in a local SQLite database. It uses
the historical sales data for training and chronological validation.

## Setup

From the project root, create a Python virtual environment and install the
dependencies. These commands use Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

1. Open the project folder in VS Code.
2. Open the project notebook under `notebooks/` and select `.venv` as the
   Jupyter kernel.
3. Run the notebook in order through the model-export cell in section 18.3.
   This creates the SQLite database, trains the models, and generates the
   example schedule and saved Ridge model.
4. Follow [Run the batch demo](#run-the-batch-demo), then execute the
   verification cell in section 18.4 to compare the two forecast CSVs.

## Model evaluation

Models were evaluated across three chronological 42-day windows. The table
reports mean absolute error (MAE) per open store-day for the same 934 stores
with complete coverage.

| Validation window | Store-weekday baseline | Ridge | Target-encoded random forest | Ridge with Christmas feature |
| --- | ---: | ---: | ---: | ---: |
| Nov–Dec 2014 | 2,105.63 | 1,526.01 | 1,760.43 | **1,405.95** |
| Apr–May 2015 | 1,213.88 | 1,079.17 | **956.69** | 1,077.93 |
| Jun–Jul 2015 | 1,260.55 | 993.25 | **892.66** | 992.82 |

The random forest performed best in spring and summer. The Ridge model with
a pre-Christmas indicator performed best in winter and was selected for the
batch demo. The winter result is exploratory because the forecast errors in
that window informed the feature design.

## Run the batch demo

Run the notebook through section 18.3 to create the example schedule and
export the trained model. Then run this command from the project root:

```powershell
python forecast.py --model data/batch_demo/ridge_winter_demo.joblib --input data/batch_demo/forecast_input.csv --output data/batch_demo/forecast_cli.csv
```

The output contains `Store`, `Date`, and `PredictedSales`. Closed store-days
receive a forecast of zero.