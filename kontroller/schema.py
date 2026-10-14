import sys

import pandas as pd
import pandera.pandas as pa

# Inspect data/kundbas.csv to find the exact column names and expected types
schema = pa.DataFrameSchema(
    {

        "kund_id": pa.Column(int),
        "alder": pa.Column(float, nullable=True),
        "region": pa.Column(str),
        "abonnemang": pa.Column(str),
        "manader_som_kund": pa.Column(int),
        "manadskostnad": pa.Column(float),
        "data_gb_per_manad": pa.Column(float, pa.Check.le(100.0)),
        "supportarenden_12m": pa.Column(int),
        "avslutat": pa.Column(int),
    },
    strict=True,
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
