import os
import openai
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import time
from src.constants.prompts import prompt_templates

api_key = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

dataset = load_dataset("glue", "sst2")
texts = dataset["validation"]["sentence"]
y_true = dataset["validation"]["label"]

results = []

for variant, template in prompt_templates.items():
    y_pred = []
    
    for text in tqdm(texts, desc=variant):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": template.format(review_text=text)}],
            temperature=0.0
        )
        ans = response.choices[0].message.content.lower()
        
        # Parsing
        if "positive" in ans: y_pred.append(1)
        elif "negative" in ans: y_pred.append(0)
        else: y_pred.append(-1)
        
df_results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
df_valid = df_results[df_results['y_pred'] != -1]

accuracy = accuracy_score(df_valid['y_true'], df_valid['y_pred'])
f1 = f1_score(df_valid['y_true'], df_valid['y_pred'], average='macro')
precision = precision_score(df_valid['y_true'], df_valid['y_pred'], average='macro')
recall = recall_score(df_valid['y_true'], df_valid['y_pred'], average='macro')

print(f"Metrics for {variant}:")
print(f"Valid Responses: {len(df_valid)}/872")
print(f"Accuracy: {accuracy*100:.2f}")
print(f"F1-Score: {f1*100:.2f}")
print(f"Precision: {precision*100:.2f}")
print(f"Recall: {recall*100:.2f}")