from openai import OpenAI
import json
import pandas as pd
import time
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load your structured JSON
with open('structured_quality_manual.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Prepare CSV file with headers
csv_filename = 'synthetic_qna_dataset.csv'
df_init = pd.DataFrame(columns=['source_header', 'question', 'answer'])
df_init.to_csv(csv_filename, index=False, encoding='utf-8-sig')


for item in dataset:
    policy_text = item['intro'] + "\n\n" + item['content']
    prompt = (
        f"Read the following policy content:\n{policy_text}\n\n"
        "Generate exactly 3 diverse and conversational Q&A pairs "
        "covering different aspects of this content. Format each pair as:\n"
        "Q: [question]\nA: [answer]\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Q&A pairs for chatbot fine-tuning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        output = response.choices[0].message.content
        print(f"✅ Q&A generated for section: {item['header']}")
        
        # Parse Q&A pairs and write directly to CSV
        rows = []
        question, answer = None, None
        for line in output.split("\n"):
            if line.strip().startswith("Q:"):
                question = line.replace("Q:", "").strip()
            elif line.strip().startswith("A:") and question:
                answer = line.replace("A:", "").strip()
                rows.append({
                    "source_header": item['header'],
                    "question": question,
                    "answer": answer
                })
                question, answer = None, None  # Reset for next pair

        # Append parsed rows immediately to CSV
        if rows:
            pd.DataFrame(rows).to_csv(csv_filename, mode='a', index=False, header=False, encoding='utf-8-sig')

        time.sleep(1)  # Optional rate limiting

        

    except Exception as e:
        print(f"Failed on {item['header']}: {e}")
        
print(f"✅ Q&A dataset generated and saved to {csv_filename}")
