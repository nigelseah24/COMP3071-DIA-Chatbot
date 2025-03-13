import json
import re

# File paths (update as needed)
file_path = "C:\\Users\\PC 5\\Desktop\\COMP3071-DIA-Chatbot\\backend\\quality_manual_data_incremental.txt"
output_json_path = "structured_quality_manual.json"

# Read the data from the txt file
with open(file_path, 'r', encoding='utf-8') as file:
    raw_text = file.read()

# Split into sections based on delimiter
sections = [section.strip() for section in raw_text.split("================================================================================") if section.strip()]

structured_data = []

# Regex patterns to extract URL, Header, Intro, and Content
url_pattern = r"URL:\s*(.+)"
header_pattern = r"Header:\s*(.+)"
intro_pattern = r"Intro:\s*(.+)"
content_pattern = r"Content:\s*(.+)"  # Will extract content including newlines

for sec in sections:
    url_match = re.search(url_pattern, sec)
    header_match = re.search(header_pattern, sec)
    intro_match = re.search(intro_pattern, sec)
    content_match = re.search(content_pattern, sec, re.DOTALL)

    url = url_match.group(1).strip() if url_match else ""
    header = header_match.group(1).strip() if header_match else ""
    intro = intro_match.group(1).strip() if intro_match else ""
    content = content_match.group(1).strip() if content_match else ""

    structured_data.append({
        "url": url,
        "header": header,
        "intro": intro,
        "content": content
    })

# Save structured data into JSON file
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(structured_data, json_file, indent=4, ensure_ascii=False)

print(f"Structured dataset saved successfully to '{output_json_path}'")
