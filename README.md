# Zepto Data & AI Platform

## Module 1: Data Pipeline

This module implements a complete data pipeline using Python, Requests, BeautifulSoup, Pandas, and SQLite.

The pipeline scrapes book data from Books to Scrape, cleans and transforms the data, stores it in a normalized SQLite database, and executes SQL queries for validation and analysis.

## Project Structure

```text
zepto-ai-ml-capstone/
│
├── README.md
│
└── data_pipeline/
    ├── scraper.py
    ├── cleaned_books.csv
    ├── database.py
    ├── books.db
    ├── queries.py
    └── sql_query_outputs.txt
```

## Requirements

Python 3.10 or later is recommended.

Required Python packages:

- requests
- beautifulsoup4
- pandas

SQLite is used through Python's built-in sqlite3 module.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install requests beautifulsoup4 pandas
```

## Running the Data Pipeline

Run the scraper:

```powershell
python -u data_pipeline\scraper.py
```

The scraper collects books from three categories:

- Fiction
- Mystery
- Classics

The resulting cleaned dataset is saved as:

```text
data_pipeline/cleaned_books.csv
```

The final dataset contains 116 books across 3 categories.

## Scraping

The scraper uses Requests and BeautifulSoup to extract:

- title
- price as listed in GBP
- star rating
- availability
- category

The scraper follows pagination links for each selected category.

## Data Cleaning

The following transformations are applied:

### Price

The GBP price is converted from text to a floating-point number after removing the pound symbol.

### Rating

Star-rating text is converted to an integer:

- One = 1
- Two = 2
- Three = 3
- Four = 4
- Five = 5

### Availability

The availability text is converted into an `in_stock` boolean value.

### Missing Values

Numeric parsing failures for `price_gbp` and `rating` are handled using median imputation.

Rows with missing or unparseable title, category, availability, or stock status are dropped.

### INR Conversion

The INR price is calculated using the fixed conversion rate required by the assignment:

```text
price_inr = price_gbp * 105.50
```

No external exchange-rate API is used.

## SQLite Database

The cleaned data is stored in:

```text
data_pipeline/books.db
```

The database contains two related tables.

### categories

```text
category_id
category_name
```

### books

```text
book_id
title
price_gbp
price_inr
rating
in_stock
category_id
```

The `books.category_id` column is a foreign key referencing `categories.category_id`.

This provides a normalized relational structure.

## Creating the Database

Run:

```powershell
python -u data_pipeline\database.py
```

The script creates the SQLite database and inserts the cleaned book data.

Expected result:

```text
Categories inserted: 3
Books inserted: 116
```

## SQL Queries

The SQL query script demonstrates:

- SELECT and WHERE
- ORDER BY
- LIMIT
- DISTINCT
- IN
- BETWEEN
- JOIN

Run:

```powershell
python -u data_pipeline\queries.py
```

The SQL query strings and outputs are saved to:

```text
data_pipeline/sql_query_outputs.txt
```

## Pandas SQL Verification

The query script uses `pandas.read_sql` to load multiple SQL query results.

The SQL JOIN result is independently reproduced using `pandas.merge` on in-memory DataFrames.

The SQL JOIN and pandas merge results are compared for equivalence.

The verification output confirms:

```text
SQL JOIN and pandas.merge equivalent: True
```

## Data Pipeline Execution Order

Run the files in the following order:

```text
1. scraper.py
2. database.py
3. queries.py
```

Commands:

```powershell
python -u data_pipeline\scraper.py
python -u data_pipeline\database.py
python -u data_pipeline\queries.py
```

## Module 1 Output

The completed pipeline produces:

- 116 scraped and cleaned books
- 3 book categories
- cleaned CSV dataset
- normalized SQLite database
- SQL query strings and outputs
- SQL JOIN and pandas merge equivalence verification

## Currency Conversion

The required fixed conversion rate is:

```text
1 GBP = 105.50 INR
```

The calculation is:

```text
price_inr = price_gbp * 105.50
```

## Reproducibility

The complete database can be recreated by running:

```powershell
python -u data_pipeline\scraper.py
python -u data_pipeline\database.py
python -u data_pipeline\queries.py
```

The SQLite database is generated locally from the cleaned CSV dataset.
