import sqlite3
import pandas as pd

CSV_PATH = "data_pipeline/cleaned_books.csv"
DB_PATH = "data_pipeline/books.db"

def create_database():
    df = pd.read_csv(CSV_PATH)

    connection = sqlite3.connect(DB_PATH)

    connection.execute("PRAGMA foreign_keys = ON")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
    """)

    cursor.execute("DELETE FROM books")
    cursor.execute("DELETE FROM categories")

    categories = df["category"].drop_duplicates().tolist()

    cursor.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(category,) for category in categories]
    )

    category_map = dict(
        cursor.execute(
            "SELECT category_id, category_name FROM categories"
        ).fetchall()
    )

    df["category_id"] = df["category"].map(
        {name: category_id for category_id, name in category_map.items()}
    )

    books_data = [
        (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            int(row["category_id"])
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        books_data
    )

    connection.commit()

    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    print("DATABASE CREATED")
    print("================")
    print(f"Database: {DB_PATH}")
    print(f"Categories inserted: {category_count}")
    print(f"Books inserted: {book_count}")

    print()
    print("Categories:")
    for row in cursor.execute(
        "SELECT category_id, category_name FROM categories ORDER BY category_id"
    ):
        print(row)

    print()
    print("First 5 books:")
    for row in cursor.execute(
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
        ORDER BY book_id
        LIMIT 5
        """
    ):
        print(row)

    connection.close()

if __name__ == "__main__":
    create_database()