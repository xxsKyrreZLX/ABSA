import time
import pandas as pd
import requests
from urllib.parse import quote_plus
from datetime import datetime

# ---------- Config ----------
STEAM_APP_ID = 2012510  # Stormgate app id
LANGUAGE = "english"     # steam language filter
MAX_REVIEWS = 5000        # how many reviews to fetch
DELAY_SECONDS = 2.0       # delay between requests to avoid throttling
FILTER_MODES = ("recent", "updated")  # iterate both to cover more reviews
OUTPUT_CSV = "steam_reviews_raw_withTime.csv"   # output file name (in current folder)


def fetch_steam_reviews(appid: int,
                        max_reviews: int = 5000,
                        language: str = "english",
                        delay: float = 2.0,
                        filter_modes=("recent", "updated")) -> list:
    """
    Fetch Steam reviews using official cursor-based pagination.
    - Each page returns up to 100 reviews
    - filter can be "recent" or "updated"
    - Continues until reaching max_reviews or no more pages
    Returns a list of dicts: {source, id, text, timestamp, date_time}
    """
    reviews = []
    seen_ids = set()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for filter_mode in filter_modes:
        cursor = "*"
        empty_page_streak = 0
        while len(reviews) < max_reviews:
            encoded_cursor = quote_plus(cursor)
            url = (
                f"https://store.steampowered.com/appreviews/{appid}"
                f"?json=1&review_type=all&purchase_type=all&language={language}"
                f"&filter={filter_mode}&num_per_page=100&filter_offtopic_activity=1&cursor={encoded_cursor}"
            )
            try:
                print(f"Fetching Steam reviews - filter: {filter_mode}, cursor_start={cursor[:8]}..., total={len(reviews)}")
                r = requests.get(url, timeout=25, headers=headers)
                r.raise_for_status()
                data = r.json()
                rows = data.get('reviews', [])
                if not rows:
                    empty_page_streak += 1
                    if empty_page_streak >= 2:
                        print("No more reviews (empty twice). Switching filter or stopping.")
                        break
                    time.sleep(delay * 2)
                    continue

                empty_page_streak = 0
                new_count = 0
                for rev in rows:
                    review_id = rev.get("recommendationid") or (
                        rev.get("author", {}).get("steamid", "") + "_" + str(rev.get("timestamp_created", ""))
                    )
                    if review_id in seen_ids:
                        continue
                    seen_ids.add(review_id)
                    # Convert timestamp to readable format
                    timestamp_created = rev.get("timestamp_created")
                    readable_time = None
                    if timestamp_created:
                        try:
                            readable_time = datetime.fromtimestamp(timestamp_created).strftime("%Y-%m-%d %H:%M:%S")
                        except (ValueError, OSError):
                            readable_time = str(timestamp_created)
                    
                    reviews.append({
                        "source": "steam",
                        "id": review_id,
                        "text": rev.get("review"),
                        "timestamp": timestamp_created,
                        "date_time": readable_time
                    })
                    new_count += 1
                    if len(reviews) >= max_reviews:
                        break

                print(f"Added {new_count} new (total: {len(reviews)})")

                next_cursor = data.get('cursor')
                if not next_cursor or next_cursor == cursor:
                    print("Cursor did not advance; likely end reached for this filter.")
                    break
                cursor = next_cursor

                if len(reviews) < max_reviews:
                    time.sleep(delay)
            except Exception as e:
                print("Steam fetch error:", e)
                time.sleep(delay * 2)
                break

    print(f"Total Steam reviews fetched: {len(reviews)}")
    return reviews


def save_reviews_to_csv(reviews: list, output_csv: str) -> None:
    df = pd.DataFrame(reviews)
    if df.empty:
        print("No reviews to save.")
        return
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} reviews to {output_csv}")


def main():
    reviews = fetch_steam_reviews(
        appid=STEAM_APP_ID,
        max_reviews=MAX_REVIEWS,
        language=LANGUAGE,
        delay=DELAY_SECONDS,
        filter_modes=FILTER_MODES,
    )
    save_reviews_to_csv(reviews, OUTPUT_CSV)


if __name__ == "__main__":
    main()
