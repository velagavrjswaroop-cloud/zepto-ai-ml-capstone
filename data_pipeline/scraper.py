import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

GBP_TO_INR = 105.50

CATEGORY_URLS = {
    "Fiction": "https://books.toscrape.com/catalogue/category/books/fiction_10/index.html",
    "Mystery": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "Classics": "https://books.toscrape.com/catalogue/category/books/classics_6/index.html"
}

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def scrape_category(category, category_url):
    books = []
    current_url = category_url

    while current_url:
        response = requests.get(
            current_url,
            headers=HEADERS,
            timeout=30
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        product_cards = soup.select("article.product_pod")

        print(f"{category}: {len(product_cards)} books found")

        for book in product_cards:
            title = None
            price_gbp = None
            rating = None
            availability = None
            in_stock = None

            title_element = book.select_one("h3 a")
            if title_element:
                title = title_element.get("title")
                if not title:
                    title = title_element.get_text(strip=True)

            price_element = book.select_one("p.price_color")
            if price_element:
                price_text = price_element.get_text(strip=True)
                try:
                    price_text = price_text.replace("Â£", "").replace("£", "").strip()
                    price_gbp = float(price_text)
                except (ValueError, TypeError):
                    price_gbp = None

            rating_element = book.select_one("p.star-rating")
            if rating_element:
                rating_classes = rating_element.get("class", [])
                for rating_name, rating_value in RATING_MAP.items():
                    if rating_name in rating_classes:
                        rating = rating_value
                        break

            availability_element = book.select_one("p.availability")
            if availability_element:
                availability = availability_element.get_text(
                    " ",
                    strip=True
                )

                if "In stock" in availability:
                    in_stock = True
                elif "Out of stock" in availability:
                    in_stock = False

            books.append({
                "title": title,
                "price_gbp": price_gbp,
                "rating": rating,
                "availability": availability,
                "in_stock": in_stock,
                "category": category
            })

        next_element = soup.select_one("li.next a")

        if next_element:
            next_url = next_element.get("href")
            current_url = urljoin(current_url, next_url)
        else:
            current_url = None

    return books

def clean_data(df):
    df = df.copy()

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    price_median = df["price_gbp"].median()
    rating_median = df["rating"].median()

    df["price_gbp"] = df["price_gbp"].fillna(price_median)
    df["rating"] = df["rating"].fillna(rating_median)

    df = df.dropna(
        subset=[
            "title",
            "category",
            "availability",
            "in_stock"
        ]
    )

    df["rating"] = df["rating"].round().astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    return df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "availability",
            "category"
        ]
    ]

def main():
    all_books = []

    for category, category_url in CATEGORY_URLS.items():
        category_books = scrape_category(
            category,
            category_url
        )
        all_books.extend(category_books)

    raw_df = pd.DataFrame(all_books)

    print()
    print(f"Raw rows collected: {len(raw_df)}")

    if raw_df.empty:
        raise RuntimeError("The scraper collected zero rows.")

    df = clean_data(raw_df)

    df = df.drop_duplicates(
        subset=["title", "category"]
    ).reset_index(drop=True)

    if len(df) < 60:
        raise RuntimeError(
            f"Only {len(df)} books were collected. At least 60 are required."
        )

    if df["category"].nunique() < 3:
        raise RuntimeError(
            f"Only {df['category'].nunique()} categories were collected. At least 3 are required."
        )

    if df["price_gbp"].isna().any():
        raise RuntimeError("Some price_gbp values are still missing.")

    if df["price_inr"].isna().any():
        raise RuntimeError("Some price_inr values are still missing.")

    output_path = "data_pipeline/cleaned_books.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print("SCRAPING COMPLETED")
    print("==================")
    print(f"Total books: {len(df)}")
    print(f"Total categories: {df['category'].nunique()}")
    print()
    print("Books by category:")
    print(df["category"].value_counts())
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()
    print("Data types:")
    print(df.dtypes)
    print()
    print("Missing values:")
    print(df.isna().sum())
    print()
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print()
    print(f"Saved file: {output_path}")

if __name__ == "__main__":
    main()