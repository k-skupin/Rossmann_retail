import argparse
from pathlib import Path

import joblib
import pandas as pd


# Required CSV columns for each scheduled store-day.
forecast_input_columns = [
    "Store", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"
]


def build_forecast_features(data, reference_date):
    """Build the model's input features from the store schedule.

    Parameters
    ----------
    data : pandas.DataFrame
        Store schedule with ``Store``, ``Date``, ``Promo``,
        ``StateHoliday``, and ``SchoolHoliday`` columns. ``Date`` may
        contain datetime values or date strings.
    reference_date : pandas.Timestamp
        Date used during training as the origin for ``days_since_start``.

    Returns
    -------
    pandas.DataFrame
        One feature row per input row, with the same index. Calendar
        features are integers, and ``StateHoliday`` is stored as text.
    """
    dates = pd.to_datetime(data["Date"]).dt.normalize()

    # Flag December 17–23 in each year using only the date.
    pre_christmas_week = dates.dt.month.eq(12) & dates.dt.day.between(17, 23)

    return pd.DataFrame({
        "Store": data["Store"],
        # Pandas starts at 0 for Monday. Rossmann starts at 1.
        "DayOfWeek": dates.dt.day_of_week + 1,
        "month": dates.dt.month,
        "StateHoliday": data["StateHoliday"].astype("string"),
        # Measure time from the same reference date used during training.
        "days_since_start": (dates - reference_date).dt.days,
        "Promo": data["Promo"],
        "SchoolHoliday": data["SchoolHoliday"],
        "pre_christmas_week": pre_christmas_week.astype("int8"),
    })



def read_schedule(path):
    """Read a store schedule from a CSV file.

    Parameters
    ----------
    path : str or pathlib.Path

        Path to the input CSV file.

    Returns
    -------
    pandas.DataFrame
        Store schedule with ``Date`` parsed as datetime and
        ``StateHoliday`` read as text.
    """
    # Read holiday codes such as "0" and "c" consistently as text.
    schedule = pd.read_csv(path, dtype={"StateHoliday": "string"})
    schedule["Date"] = pd.to_datetime(schedule["Date"], format="%Y-%m-%d")
    return schedule


def predict_batch(schedule, model, reference_date):
    """Predict daily sales for each row of the store schedule.

    Parameters
    ----------
    schedule : pandas.DataFrame
        Store schedule with ``Store``, ``Date``, ``Open``, ``Promo``,
        ``StateHoliday``, and ``SchoolHoliday`` columns. ``Open`` uses
        1 for open store-days and 0 for closed store-days.
    model : sklearn.pipeline.Pipeline
        Fitted regression pipeline that accepts the columns returned
        by ``build_forecast_features()``.
    reference_date : pandas.Timestamp
        Date used during training as the origin for ``days_since_start``.

    Returns
    -------
    pandas.DataFrame
        One row per input row, in the original order. ``Store`` and
        ``Date`` are copied from the schedule. ``PredictedSales`` contains
        non-negative floating-point forecasts, with 0.0 for closed stores.

    Raises
    ------
    ValueError
        If ``Open`` contains values other than 0 or 1.
    """
    schedule = schedule[forecast_input_columns]

    # Validate the opening status before deciding which rows need predictions.
    if not schedule["Open"].isin([0, 1]).all():
        raise ValueError("Open must contain only 0 or 1.")

    open_rows = schedule["Open"].eq(1)

    # Start with zero sales for every row, including closed store-days.
    result = schedule[["Store", "Date"]].copy()
    result["PredictedSales"] = 0.0

    # Run the model only for open store-days.
    if open_rows.any():
        features = build_forecast_features(
            schedule.loc[open_rows], reference_date
        )
        # Clip negative forecasts to zero before writing them into the result.
        result.loc[open_rows, "PredictedSales"] = (
            model.predict(features).clip(0, None)
        )

    return result


def main():
    """Run the batch forecast from the command line.

    Returns
    -------
    None
        Writes the forecast to CSV and prints the output path.

    Notes
    -----
    The following required command-line arguments are parsed as
    ``pathlib.Path`` objects.

    - ``--model`` points to the saved model and its reference date.
    - ``--input`` points to the store schedule CSV.
    - ``--output`` specifies where to write the forecast CSV.
    """
    # Define the three required arguments and convert their values to paths.
    parser = argparse.ArgumentParser(description="Forecast daily retail sales.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Load the fitted pipeline and reference date saved by the notebook.
    artifact = joblib.load(args.model)
    schedule = read_schedule(args.input)
    forecast = predict_batch(
        schedule, artifact["model"], artifact["reference_date"]
    )

    # Create the output folder if needed and save the forecast as CSV.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(forecast)} forecasts to {args.output.resolve()}")


# Start the batch run when this file is executed from the terminal.
if __name__ == "__main__":
    main()