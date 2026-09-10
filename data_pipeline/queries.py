import sqlite3
import pandas as pd

DB_PATH = "data_pipeline/books.db"
OUTPUT_PATH = "data_pipeline/sql_query_outputs.txt"

queries = {
    "Query 1 - SELECT WHERE": """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4;
""",
    "Query 2 - ORDER BY LIMIT": """
SELECT title, price_inr
FROM books
ORDER BY price_inr DESC
LIMIT 10;
""",
    "Query 3 - DISTINCT": """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
""",
    "Query 4 - IN": """
SELECT title, rating, price_gbp
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC;
""",
    "Query 5 - BETWEEN": """
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;
""",
    "Query 6 - JOIN": """
SELECT
    b.book_id,
    b.title,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock,
    c.category_name
FROM books b
JOIN categories c
    ON b.category_id = c.category_id
ORDER BY c.category_name, b.rating DESC, b.title
LIMIT 20;
"""
}

def main():
    connection = sqlite3.connect(DB_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output:
        for query_name, query in queries.items():
            print()
            print(query_name)
            print("=" * len(query_name))
            print(query.strip())

            output.write("\n")
            output.write(query_name + "\n")
            output.write("=" * len(query_name) + "\n")
            output.write(query.strip() + "\n\n")

            result = pd.read_sql(query, connection)

            print(result.to_string(index=False))

            output.write(result.to_string(index=False))
            output.write("\n\n")

    query_1_result = pd.read_sql(
        queries["Query 1 - SELECT WHERE"],
        connection
    )

    query_2_result = pd.read_sql(
        queries["Query 2 - ORDER BY LIMIT"],
        connection
    )

    sql_join_result = pd.read_sql(
        queries["Query 6 - JOIN"],
        connection
    )

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )

    pandas_join_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_join_result = pandas_join_result[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name"
        ]
    ]

    pandas_join_result = pandas_join_result.sort_values(
        by=["category_name", "rating", "title"],
        ascending=[True, False, True]
    ).head(20).reset_index(drop=True)

    sql_join_result = sql_join_result.reset_index(drop=True)

    sql_join_result["in_stock"] = sql_join_result["in_stock"].astype(bool)
    pandas_join_result["in_stock"] = pandas_join_result["in_stock"].astype(bool)

    numeric_columns = [
        "price_gbp",
        "price_inr"
    ]

    for column in numeric_columns:
        sql_join_result[column] = sql_join_result[column].round(6)
        pandas_join_result[column] = pandas_join_result[column].round(6)

    join_equivalent = sql_join_result.equals(
        pandas_join_result
    )

    with open(OUTPUT_PATH, "a", encoding="utf-8") as output:
        output.write("\n")
        output.write("PANDAS READ_SQL VERIFICATION\n")
        output.write("============================\n")
        output.write("Query 1 loaded using pd.read_sql:\n")
        output.write(query_1_result.to_string(index=False))
        output.write("\n\n")
        output.write("Query 2 loaded using pd.read_sql:\n")
        output.write(query_2_result.to_string(index=False))
        output.write("\n\n")
        output.write("SQL JOIN RESULT:\n")
        output.write(sql_join_result.to_string(index=False))
        output.write("\n\n")
        output.write("PANDAS MERGE RESULT:\n")
        output.write(pandas_join_result.to_string(index=False))
        output.write("\n\n")
        output.write(
            f"SQL JOIN and pandas.merge equivalent: {join_equivalent}\n"
        )

    print()
    print("PANDAS VERIFICATION")
    print("===================")
    print(f"Query 1 rows read with pd.read_sql: {len(query_1_result)}")
    print(f"Query 2 rows read with pd.read_sql: {len(query_2_result)}")
    print(f"SQL JOIN rows: {len(sql_join_result)}")
    print(f"Pandas merge rows: {len(pandas_join_result)}")
    print(f"SQL JOIN and pandas.merge equivalent: {join_equivalent}")
    print()
    print(f"Saved query strings and outputs: {OUTPUT_PATH}")

    connection.close()

if __name__ == "__main__":
    main()