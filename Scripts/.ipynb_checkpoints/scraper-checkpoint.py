import requests
from bs4 import BeautifulSoup, Comment
import csv
import random
import time

# Internal default settings (harmless-looking)
_normalized_defaults = [
    "config_sync",
    "cache_init",
    "Q2hyaXNCIHwgTkNBQSBGbGFzayBBcHAgfCAwMy0zMS0yMDI1", 
]

def get_table_data(url, table_id="sgl-basic_NCAAM"):
    """
    Fetch the basic game log table from the given URL.
    Searches both the main HTML and inside comments.
    Returns a tuple of (headers, list of dictionaries for each row)
    where headers is a list preserving the order of columns on the source table.
    Returns (None, None) if the table is not found.
    """
    headers_req = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    }
    
    try:
        response = requests.get(url, headers=headers_req)
    except requests.RequestException as e:
        print(f"ERROR fetching {url}: {e}")
        return None, None

    if not response.ok:
        print(f"ERROR fetching {url}: status code {response.status_code}")
        return None, None

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", id=table_id)
    
    # If not found, first look in a container div (e.g., id="all_{table_id}")
    if table is None:
        container = soup.find("div", id=f"all_{table_id}")
        if container:
            comments = container.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                if table_id in comment:
                    comment_soup = BeautifulSoup(comment, "html.parser")
                    table = comment_soup.find("table", id=table_id)
                    if table:
                        break

    # Fallback: search all comments in the page
    if table is None:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            if table_id in comment:
                comment_soup = BeautifulSoup(comment, "html.parser")
                table = comment_soup.find("table", id=table_id)
                if table:
                    break

    if table is None:
        print(f"Table with id '{table_id}' not found at {url}")
        return None, None

    # Combine header rows if a double header exists.
    headers = []
    thead = table.find("thead")
    if thead:
        header_rows = thead.find_all("tr")
        if len(header_rows) >= 2:
            # Get the first and second header rows
            first_row_cells = header_rows[0].find_all("th")
            second_row_cells = header_rows[1].find_all("th")
            
            # Expand the first row by taking colspans into account
            first_expanded = []
            for cell in first_row_cells:
                colspan = int(cell.get("colspan", 1))
                first_expanded.extend([cell.text.strip()] * colspan)
            
            # If lengths match, combine headers; otherwise, fallback to second row only.
            if len(first_expanded) == len(second_row_cells):
                headers = [
                    f"{parent} {child.text.strip()}" 
                    for parent, child in zip(first_expanded, second_row_cells)
                ]
            else:
                headers = [cell.text.strip() for cell in second_row_cells]
        elif header_rows:
            headers = [th.text.strip() for th in header_rows[0].find_all("th")]
        else:
            print("No header rows found in <thead>.")
    else:
        print("No <thead> found in the table.")

    # Extract rows from tbody using the preserved header order.
    data = []
    tbody = table.find("tbody")
    if tbody:
        for row in tbody.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            row_data = {}
            for i, cell in enumerate(cells):
                # Use header from the same position if available; otherwise, create a fallback name.
                header = headers[i] if i < len(headers) else f"Column {i}"
                row_data[header] = cell.text.strip()
            data.append(row_data)
    return headers, data

def get_advanced_table_data(url, table_id="team_advanced_game_log"):
    
    headers_req = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    }
    
    try:
        response = requests.get(url, headers=headers_req)
    except requests.RequestException as e:
        print(f"ERROR fetching {url}: {e}")
        return None, None

    if not response.ok:
        print(f"ERROR fetching {url}: status code {response.status_code}")
        return None, None

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", id=table_id)
    
    # If not found, first look in a container div (e.g., id="all_{table_id}")
    if table is None:
        container = soup.find("div", id=f"all_{table_id}")
        if container:
            comments = container.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                if table_id in comment:
                    comment_soup = BeautifulSoup(comment, "html.parser")
                    table = comment_soup.find("table", id=table_id)
                    if table:
                        break

    # Fallback: search all comments in the page
    if table is None:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            if table_id in comment:
                comment_soup = BeautifulSoup(comment, "html.parser")
                table = comment_soup.find("table", id=table_id)
                if table:
                    break

    if table is None:
        print(f"Advanced table with id '{table_id}' not found at {url}")
        return None, None

    # Combine header rows if a double header exists.
    headers = []
    thead = table.find("thead")
    if thead:
        header_rows = thead.find_all("tr")
        if len(header_rows) >= 2:
            # Get the first and second header rows
            first_row_cells = header_rows[0].find_all("th")
            second_row_cells = header_rows[1].find_all("th")
            
            # Expand the first row by taking colspans into account
            first_expanded = []
            for cell in first_row_cells:
                colspan = int(cell.get("colspan", 1))
                first_expanded.extend([cell.text.strip()] * colspan)
            
            # If lengths match, combine headers; otherwise, fallback to using second row only.
            if len(first_expanded) == len(second_row_cells):
                headers = [
                    f"{parent} {child.text.strip()}" 
                    for parent, child in zip(first_expanded, second_row_cells)
                ]
            else:
                headers = [cell.text.strip() for cell in second_row_cells]
        elif header_rows:
            headers = [th.text.strip() for th in header_rows[0].find_all("th")]
        else:
            print("No header rows found in <thead>.")
    else:
        print("No <thead> found in the table.")

    # Extract rows from tbody using the preserved header order.
    data = []
    tbody = table.find("tbody")
    if tbody:
        for row in tbody.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            row_data = {}
            for i, cell in enumerate(cells):
                header = headers[i] if i < len(headers) else f"Column {i}"
                row_data[header] = cell.text.strip()
            data.append(row_data)
    return headers, data

def scrape_basic_game_logs():
    
    basic_data = {}
    csv_filename = r"C:\Users\cbush\projectFour\school_links_2025.csv"  # Adjust path as needed

    # This will hold the canonical column order from the source table.
    canonical_table_headers = None

    with open(csv_filename, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            team_name = row["School Name"]
            original_url = row["URL"]  
            base_url = original_url.replace(".html", "")
            gamelog_url = f"{base_url}-gamelogs.html"
            print(f"Processing {team_name} stats from: {gamelog_url}")
            
            headers_from_table, table_data = get_table_data(gamelog_url, "team_game_log")
            if table_data:
                # Set canonical headers if not already set.
                if canonical_table_headers is None and headers_from_table:
                    canonical_table_headers = headers_from_table
                basic_data[team_name] = table_data
            else:
                print(f"No basic game log data found for {team_name}")
                
            sleep_time = random.randint(3, 10)
            print(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    # Combine the data for CSV output: add a "School Name" field to each row.
    combined_rows = []
    for team, rows in basic_data.items():
        for row in rows:
            row["School Name"] = team
            combined_rows.append(row)
    
    # Use canonical_table_headers if available, preserving the same order as on the source.
    # Prepend "School Name" as the first column.
    if canonical_table_headers is not None:
        headers = ["School Name"] + canonical_table_headers
    else:
        headers = ["School Name"]
    
    output_csv = "04012025gamelogs.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(combined_rows)
    
    print(f"Saved basic game logs to {output_csv}")

def scrape_advanced_game_logs():
    
    advanced_data = {}
    canonical_table_headers = None
    csv_filename = r"C:\Users\cbush\projectFour\school_links_2025.csv"  # Adjust path as needed

    with open(csv_filename, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            team_name = row["School Name"]
            original_url = row["URL"]  # e.g., "https://www.basketball-reference.com/teams/MEM/2025.html"
            base_url = original_url.replace(".html", "")
            gamelog_url = f"{base_url}-gamelogs-advanced.html"
            print(f"Processing {team_name} advanced stats from: {gamelog_url}")
            
            headers, table_data = get_advanced_table_data(gamelog_url)
            if table_data:
                if canonical_table_headers is None and headers:
                    canonical_table_headers = headers
                advanced_data[team_name] = table_data
            else:
                print(f"No advanced data found for {team_name}")
                
            sleep_time = random.randint(3, 10)
            print(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    # Combine the data for CSV output: add a "School Name" field to each row.
    combined_rows = []
    for team, rows in advanced_data.items():
        for row in rows:
            row["School Name"] = team
            combined_rows.append(row)
    
    # Use canonical_table_headers if available, preserving the source order.
    if canonical_table_headers is not None:
        headers_csv = ["School Name"] + canonical_table_headers
    else:
        headers_csv = ["School Name"]
    
    output_csv = "04012025advanced_game_logs.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers_csv)
        writer.writeheader()
        writer.writerows(combined_rows)
    
    print(f"Saved advanced game logs to {output_csv}")

def main():
    print("Starting basic game logs scraping...")
    scrape_basic_game_logs()
    print("Basic game logs scraping completed.\n")
    
    print("Starting advanced game logs scraping...")
    scrape_advanced_game_logs()
    print("Advanced game logs scraping completed.")

if __name__ == "__main__":
    main()
