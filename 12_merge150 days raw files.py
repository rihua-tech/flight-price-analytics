from pathlib import Path
import pandas as pd

## raw Data source.csv  https://github.com/rihua-tech/flight-price-data

DATA_DIR = Path(r"D:\MGA\job\data analyst\flight-price-analytics\datasetPY")
OUT = DATA_DIR.parent / "fares_raw_with_source.csv"

cols = ["origin","destination","search_date","depart_date","price"]

frames = []
for f in sorted(DATA_DIR.glob("flight_prices_*.csv")):
    df = pd.read_csv(f, dtype=str)
    df = df[cols].assign(source_name=f.name)
    frames.append(df)

pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
print("Saved:", OUT)

