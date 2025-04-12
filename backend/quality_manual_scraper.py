from collections import deque
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import re

def extract_clean_text(url):
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = '\n'.join(lines)
    else:
        print(f'Failed to retrieve content: {response.status_code}')
        return "", "", ""

    # Remove header and footer patterns (example logic; adjust as needed)
    pattern_without_header = r"\nUK\nChina\nMalaysia\nMain Menu\n.*?Covid-19\nPolicy A-Z\nA-Z\n"
    text_without_header = re.sub(pattern_without_header, "", cleaned_text, flags=re.DOTALL)

    pattern_without_search = r"\nSearch the manual.*?\nSearch\n"
    text_without_header_and_search = re.sub(pattern_without_search, "", text_without_header, flags=re.DOTALL)

    pattern_without_footer = (
        r"\nPolicies A-Z\nGovernance\nRecent changes.*?\nCampus maps\n|\nMore contact information\n|\nJobs\n"
        r"Browser does not support script.\nBrowser does not support script."
    )
    final_text = re.sub(pattern_without_footer, "", text_without_header_and_search, flags=re.DOTALL)

    # Extract an intro paragraph and header from the page
    header_text = ""
    try:
        page_response = requests.get(url, timeout=10)
        if page_response.status_code == 200:
            soup = BeautifulSoup(page_response.text, "html.parser")

            div_tag = soup.find('div', class_='sys_one_7030')
            if div_tag:
                h1_tag = div_tag.find('h1')
                if h1_tag:
                    header_text = h1_tag.get_text(strip=True)
                    print("Header text: " + header_text)
                else:
                    print("No <h1> tag found within the specified <div>.")
            else:
                print("No <div> with class 'sys_one_7030' found.")

            # Extract intro paragraph using BeautifulSoup
            intro_paragraph = soup.find("p", class_="introParagraph")
            intro_text = intro_paragraph.get_text(strip=True) if intro_paragraph else "No intro paragraph found"
        else:
            print(f"Failed to fetch HTML from {url}, Status Code: {page_response.status_code}")
            intro_text = "No intro paragraph found"
    except Exception as e:
        print(f"Error extracting intro paragraph from {url}: {str(e)}")
        intro_text = "No intro paragraph found"

    return intro_text, final_text, header_text

def save_page_to_file(page_info, filename):
    """
    Appends a single page's data to the specified file.
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"URL: {page_info['url']}\n")
        f.write(f"Header: {page_info['header_text']}\n")
        f.write(f"Intro: {page_info['intro_text']}\n")
        f.write("Content:\n")
        f.write(page_info['final_text'])
        f.write("\n\n" + "="*80 + "\n\n")

def crawl_quality_manual(start_url, output_file):
    visited = set()
    queue = deque([start_url])
    results = []  # optional: if you want to return all results as well

    # Optionally, clear the file at the start so we write fresh data
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Quality Manual Crawl Results ===\n\n")

    while queue:
        current_url = queue.popleft()
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            response = requests.get(current_url, timeout=10)
            if response.status_code == 200:
                # 1) Extract the cleaned content
                intro_text, final_text, header_text = extract_clean_text(current_url)

                # Create a dictionary of the scraped data
                page_info = {
                    "url": current_url,
                    "intro_text": intro_text,
                    "final_text": final_text,
                    "header_text": header_text
                }

                # 2) Immediately save the result to file
                save_page_to_file(page_info, output_file)

                # Also add to results in memory if desired
                results.append(page_info)

                # 3) Parse the page to find new links to crawl
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href:
                        absolute_url = urljoin(current_url, href)
                        # Only follow links under qualitymanual, skip "index.aspx", "a-z.aspx", and skip any .xml files
                        if (
                            absolute_url.startswith("https://www.nottingham.ac.uk/qualitymanual/") and
                            "index.aspx" not in absolute_url.lower() and
                            "a-z.aspx" not in absolute_url.lower() and
                            "quality-manual.aspx" not in absolute_url.lower() and
                            not absolute_url.lower().endswith(".xml") and not absolute_url.lower().endswith(".pdf") and not absolute_url.lower().endswith(".docx") and not absolute_url.lower().endswith(".doc")
                        ):
                            if absolute_url not in visited:
                                queue.append(absolute_url)

                # 4) Introduce a 1-second delay
                time.sleep(1)
        except Exception as e:
            print(f"Error fetching {current_url}: {e}")

    # If you only want to store data in the file, you can skip returning anything
    return results  # optional

if __name__ == "__main__":
    start_url = "https://www.nottingham.ac.uk/qualitymanual/coming-soon.aspx"
    output_file = "quality_manual_data_incremental.txt"

    data = crawl_quality_manual(start_url, output_file)
    print("Crawl finished. Data is continuously saved in", output_file)
