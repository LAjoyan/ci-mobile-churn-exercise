import sys

import pandas as pd
import pandera as pa

# Inspect data/kundbas.csv to find the exact column names and expected types
schema = pa.DataFrameSchema(
    {
        # Använd namnet från den ursprungliga kundbas.csv
        "Data_gb_per_manad": pa.Column(float, pa.Check.le(100.0)),
        # Define the rest of your columns here
        # Du kan även lägga till andra kolumner för en mer komplett validering, till exempel:
        "Kund_id": pa.Column(int),
        "Alder": pa.Column(int),
        "Region": pa.Column(str),
        "Avslutat": pa.Column(int),
    }
)


def validate_data(file_path):
    df = pd.read_csv(file_path)
    try:
        schema.validate(df, lazy=True)
        print(f"Validation successful for {file_path}")
    except pa.errors.SchemaErrors as err:
        print(f"Schema errors found in {file_path}:")
        print(err.failure_cases)
        sys.exit(1)  # Exits with an error code to fail the CI pipeline


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "data/kundbas.csv"
    validate_data(target_file)
