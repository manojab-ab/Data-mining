import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
parquet_pattern = str((BASE_DIR / "object_store" / "store_id=*" / "year=*" / "month=*" / "*.parquet").as_posix())

con = duckdb.connect()

query = f"""
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT md5(
        CAST(bill_no AS VARCHAR) || '|' || CAST(line_no AS VARCHAR)
    )) AS unique_lines
FROM read_parquet(
    '{parquet_pattern}',
    hive_partitioning=true
)
"""

result = con.execute(query).fetchone()

print("=" * 60)
print("PARQUET DUPLICATE CHECK")
print("=" * 60)
print("Total rows :", result[0])
print("Unique lines:", result[1])
print("=" * 60)

con.close()