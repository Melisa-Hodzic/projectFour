import os
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.abspath("Scripts"))

import scraper
import cleaner
import averager


TIMESTAMP_FILE = "last_scrape.txt"

def should_scrape():
    if not os.path.exists(TIMESTAMP_FILE):
        return True
    with open(TIMESTAMP_FILE, "r") as f:
        last_run = datetime.strptime(f.read().strip(), "%Y-%m-%d %H:%M:%S")
        return datetime.now() - last_run > timedelta(days=7)

def update_timestamp():
    with open(TIMESTAMP_FILE, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if should_scrape():
    print(">>> Running scraper, cleaner, and averager...")
    scraper.main()
    cleaner.main()
    averager.main()
    update_timestamp()
else:
    print(">>> Skipping scraper. Already ran this week.")
