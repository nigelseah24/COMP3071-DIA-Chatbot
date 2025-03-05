from openai import OpenAI
import os
import csv
from dotenv import load_dotenv

# Load environment variables (ensure your .env file contains OPENAI_API_KEY)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Example data with section headers
sections = [
    {
        "section_header": "1.0 Purpose",
        "context": """
        This page is to help staff who are designing new modules as well as those who are updating existing module specifications.
        It provides the regulations about how to complete module specifications.
        Module specification content is published to both applicants and current students in the Online Curriculum catalogue.
        """
    },
    {
        "section_header": "2.0 Admissions Policy",
        "context": """
        All Schools will admit students in line with the University's admissions policy. Entry criteria are set by individual Schools 
        and should be transparent and justifiable.
        Consideration will be given as to whether applicants will be able to fulfil the objectives of the programme of study and achieve the standards required.
        """
    }
]


# Function to generate structured Q&A pairs
def generate_qa_pairs(section, num_questions=2):
    prompt = f"""
    Read the following section from a university quality manual and generate {num_questions} question-answer pairs.

    Section Header: {section['section_header']}
    Context:
    {section['context']}

    Format the output as a CSV row with:
    "question","section_header","context","answer"

    Example output:
    "What is the purpose of this page?","1.0 Purpose","This page is to help staff who are designing new modules as well as those who are updating existing module specifications.","The page provides regulations for designing and updating module specifications."
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=500
    )
    
    # Extract generated Q&A
    qa_pairs = response.choices[0].message.content.strip().split("\n")

    # Convert to structured list
    structured_qa = []
    for row in qa_pairs:
        fields = row.strip('"').split('","')  # Split by CSV format
        if len(fields) == 4:
            structured_qa.append({
                "question": fields[0],
                "section_header": fields[1],
                "context": fields[2],
                "answer": fields[3]
            })
    
    return structured_qa

# Generate dataset and save to CSV
csv_filename = "qa_dataset.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["question", "section_header", "context", "answer"])  # CSV header

    for section in sections:
        qa_pairs = generate_qa_pairs(section, num_questions=3)
        for qa in qa_pairs:
            writer.writerow([qa["question"], qa["section_header"], qa["context"], qa["answer"]])

print(f"\n✅ Synthetic Q&A dataset saved to '{csv_filename}'!")