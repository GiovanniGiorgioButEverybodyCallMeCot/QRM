from __future__ import annotations
from pathlib import Path
import pandas as pd


def create_TotalReturnIndices(
    excel_path: str,
    chosen_assets: list[str],
    source_sheet_name: str = "RI",
    output_sheet_name: str = "TotalReturnIndices",
) -> None:
    """
    Read the given Excel sheet, normalize the first column dates to ISO (YYYY-MM-DD),
    remove the suffix " - TOT RETURN IND" from all column headers, and add the
    result as a new sheet named "TotalReturnIndices" to the same Excel file.

    Args:
        excel_path: Path to the source .xlsx file.
        chosen_assets: List of asset names to process.
        sheet_name: Name of the worksheet to process.
    """

    input_path = Path(excel_path)

    assert input_path.exists(), f"Input Excel file not found: {input_path}"

    used_columns = ["DATE"] + chosen_assets
    df = pd.read_excel(input_path, sheet_name=source_sheet_name, usecols=used_columns)

    # Convert first column to ISO date format
    parsed_dates = pd.to_datetime(df["DATE"], errors="coerce")
    df["DATE"] = parsed_dates.dt.strftime("%Y-%m-%d")

    # Clean column names
    cleaned_columns = [str(col).replace(" - TOT RETURN IND", "") for col in df.columns]
    df.columns = cleaned_columns

    # Add the modified sheet to the existing Excel file
    # The output sheet is correctly hard-coded here as per your docstring
    with pd.ExcelWriter(input_path, engine="openpyxl", mode="a") as writer:
        df.to_excel(writer, index=False, sheet_name=output_sheet_name)
