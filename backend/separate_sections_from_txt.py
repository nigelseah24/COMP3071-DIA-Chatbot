import os
import re

# Load the text
with open('backend\\postgraduate_regulations.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Split by the delimiter
sections = content.split("================================================================================")

# Create output directory
output_dir = 'postgraduate regulations files'
os.makedirs(output_dir, exist_ok=True)

for i, section in enumerate(sections):
    section = section.strip()
    if not section:
        continue

    # Extract the Header line
    header_match = re.search(r'Header:\s*(.*)', section)
    if header_match:
        header = header_match.group(1).strip()
        # Clean the header to make it filename-safe
        safe_header = re.sub(r'[\\/*?:"<>|]', "_", header)
        filename = f"{safe_header}.txt"
    else:
        # Fallback if no header is found
        filename = f"section_{i+1}.txt"

    # Save the section
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(section)
    print(f"✅ Saved: {filepath}")

print(f"✅ All sections saved in '{output_dir}' folder")
