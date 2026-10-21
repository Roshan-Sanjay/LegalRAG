import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

from configs.scraper_queries import LEGAL_SEARCH_QUERIES


BASE_URL = "https://indiankanoon.org"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# -------------------------------
# STEP 1: GET SEARCH PAGE LINKS
# -------------------------------
def get_case_links(search_url: str, limit: int = 20):

    print(f"\nFetching search page:\n{search_url}\n")

    response = requests.get(search_url, headers=HEADERS)

    print("Search page status:", response.status_code)

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "lxml")

    # DEBUG: confirm page is correct
    print("\nPage title:", soup.title.text if soup.title else "No title")

    links = []

    # STRATEGY 1: extract all /doc/ links (MOST RELIABLE)
    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "/doc/" in href:
            full_url = BASE_URL + href
            if full_url not in links:
                links.append(full_url)

        if len(links) >= limit:
            break

    print(f"\nExtracted {len(links)} case links")

    return links


# -------------------------------
# STEP 2: SCRAPE SINGLE CASE
# -------------------------------
def scrape_case(case_url: str):

    print(f"\nScraping: {case_url}")

    response = requests.get(case_url, headers=HEADERS)

    print("Status:", response.status_code)

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "lxml")

    # -----------------------
    # TITLE EXTRACTION
    # -----------------------
    title = soup.title.text.strip() if soup.title else "Unknown Case"

    bad_keywords = [
        "section",
        "article",
        "constitution",
        "act",
        "rules"
    ]
    
    if any(word in title.lower() for word in bad_keywords):
        return None

    # Clean title
    title = re.sub(r"\s+", " ", title)

    # -----------------------
    # TEXT EXTRACTION STRATEGY (ROBUST)
    # -----------------------

    text = ""

    # Strategy 1: known judgment containers
    selectors = [
        "div.judgments",
        "div.judgment",
        "div#judgments",
        "pre"
    ]

    for sel in selectors:
        tag = soup.select_one(sel)
        if tag:
            text = tag.get_text(" ", strip=True)
            break

    # Strategy 2: fallback full page text (LAST RESORT)
    if not text or len(text) < 200:
        text = soup.get_text(" ", strip=True)

    # -----------------------
    # CLEANING
    # -----------------------
    text = re.sub(r"\s+", " ", text)

    if len(text) < 300:
        print("Skipping (too short text)")
        return None

    return {
        "case_title": title,
        "source": case_url,
        "text": text
    }


# -------------------------------
# STEP 3: MAIN PIPELINE
# -------------------------------
def main():

    dataset = []

    visited_links = set()

    for query in LEGAL_SEARCH_QUERIES:

        print(f"\n========== QUERY: {query} ==========\n")

        search_url = (
            f"https://indiankanoon.org/search/?formInput={query}"
        )

        links = get_case_links(search_url, limit=10)

        print(f"Found {len(links)} links")

        for link in links:

            if link in visited_links:
                continue

            visited_links.add(link)

            try:

                case_data = scrape_case(link)

                if case_data:
                    dataset.append(case_data)

            except Exception as e:
                print("Error:", e)

            time.sleep(1.5)

    print(f"\nFinal dataset size: {len(dataset)}")

    df = pd.DataFrame(dataset)

    df.to_csv("data/raw/judgments.csv", index=False)

    print("\nDataset saved successfully.")

if __name__ == "__main__":
    main()