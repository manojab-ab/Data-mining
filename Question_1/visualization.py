import duckdb
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIS_DIR = BASE_DIR / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

parquet_pattern = str((BASE_DIR / "object_store" / "store_id=*" / "year=*" / "month=*" / "*.parquet").as_posix())

con = duckdb.connect()

# ============================================================
# 1. MONTHLY REVENUE
# ============================================================

monthly = con.execute(f"""
SELECT
    STRFTIME(business_date, '%Y-%m') AS month,
    ROUND(SUM(revenue), 2) AS revenue
FROM read_parquet(
    '{parquet_pattern}',
    hive_partitioning=true
)
GROUP BY 1
ORDER BY 1
""").fetchall()

months = [row[0] for row in monthly]
monthly_revenue = [row[1] for row in monthly]

plt.figure(figsize=(12, 6))
plt.plot(months, monthly_revenue, marker="o")
plt.title("Monthly Revenue - 2024")
plt.xlabel("Month")
plt.ylabel("Revenue (INR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(VIS_DIR / "monthly_revenue.png", dpi=300)
plt.close()

# ============================================================
# 2. REVENUE BY STORE
# ============================================================

store = con.execute(f"""
SELECT
    store_id,
    ROUND(SUM(revenue), 2) AS revenue
FROM read_parquet(
    '{parquet_pattern}',
    hive_partitioning=true
)
GROUP BY store_id
ORDER BY store_id
""").fetchall()

stores = [row[0] for row in store]
store_revenue = [row[1] for row in store]

plt.figure(figsize=(12, 6))
plt.bar(stores, store_revenue)
plt.title("Revenue by Store - 2024")
plt.xlabel("Store")
plt.ylabel("Revenue (INR)")
plt.tight_layout()
plt.savefig(VIS_DIR / "revenue_by_store.png", dpi=300)
plt.close()

# ============================================================
# 3. REVENUE BY CATEGORY
# ============================================================

con.execute("INSTALL postgres;")
con.execute("LOAD postgres;")

con.execute("""
ATTACH 'host=localhost port=5432 dbname=annapurna user=annapurna password=annapurna'
AS pg (TYPE POSTGRES, READ_ONLY)
""")

category = con.execute(f"""
SELECT
    c.category_name,
    ROUND(SUM(s.revenue), 2) AS revenue
FROM read_parquet(
    '{parquet_pattern}',
    hive_partitioning=true
) s
JOIN pg.public.products p
    ON s.product_code = p.product_code
   AND s.business_date >= p.valid_from
   AND s.business_date <= p.valid_to
JOIN pg.public.product_categories c
    ON p.category_id = c.category_id
GROUP BY c.category_name
ORDER BY revenue DESC
""").fetchall()

categories = [row[0] for row in category]
category_revenue = [row[1] for row in category]

plt.figure(figsize=(12, 7))
plt.barh(categories, category_revenue)
plt.title("Revenue by Category - 2024")
plt.xlabel("Revenue (INR)")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(VIS_DIR / "revenue_by_category.png", dpi=300)
plt.close()

con.close()

print("=" * 70)
print("VISUALIZATION FILES CREATED")
print("=" * 70)
print(f"1. {VIS_DIR / 'monthly_revenue.png'}")
print(f"2. {VIS_DIR / 'revenue_by_store.png'}")
print(f"3. {VIS_DIR / 'revenue_by_category.png'}")
print("=" * 70)