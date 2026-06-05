import os
import time
import datetime
import numpy as np
import pandas as pd
import ollama
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

start_total = time.time()

dataset = load_dataset("nyu-mll/glue", "sst2")
texts_to_classify = dataset["validation"]["sentence"]
y_true_llm = np.array(dataset["validation"]["label"])

output_dir = "ollama_artifacts"
os.makedirs(output_dir, exist_ok=True)
from src.constants.prompts import prompt_templates

metrics_summary = []

for variant_name, template in prompt_templates.items():
    print(f"\n========================================")
    print(f"Local inferency: {variant_name}")
    start_variant = time.time()
    
    y_pred_llm = []
    y_pred_probs_llm = []
    
    # more tokens for CoT reasoning
    max_tokens_limit = 200 if "CoT" in variant_name else 10
    
    for text in tqdm(texts_to_classify, desc=variant_name):
        prompt = template.format(review_text=text)
        
        try:
            # local call using Ollama
            response = ollama.chat(model='llama3.1:8b', messages=[
                {'role': 'user', 'content': prompt}
            ], options={
                'temperature': 0.0,
                'num_predict': max_tokens_limit 
            })
            
            response_text = response['message']['content'].strip().lower()
            
            if "CoT" in variant_name:
                if "final classification: positive" in response_text:
                    y_pred_llm.append(1)
                    y_pred_probs_llm.append(1.0)
                elif "final classification: negative" in response_text:
                    y_pred_llm.append(0)
                    y_pred_probs_llm.append(0.0)
                else:
                    y_pred_llm.append(-1)
                    y_pred_probs_llm.append(0.5)
            else:
                if "positive" in response_text and "negative" not in response_text:
                    y_pred_llm.append(1)
                    y_pred_probs_llm.append(1.0)
                elif "negative" in response_text and "positive" not in response_text:
                    y_pred_llm.append(0)
                    y_pred_probs_llm.append(0.0)
                else:
                    y_pred_llm.append(-1) 
                    y_pred_probs_llm.append(0.5)
                
        except Exception as e:
            print(f"Error: {e}")
            y_pred_llm.append(-1)
            y_pred_probs_llm.append(0.5)

    end_variant = time.time()
    tempo_variant = datetime.timedelta(seconds=int(end_variant - start_variant))

    valid_indices = [i for i, val in enumerate(y_pred_llm) if val != -1]
    y_true_valid = y_true_llm[valid_indices]
    y_pred_valid = np.array(y_pred_llm)[valid_indices]
    y_probs_valid = np.array(y_pred_probs_llm)[valid_indices]
    
    if len(valid_indices) == 0:
        print(f"Total failure in parsing for {variant_name}.")
        continue
        
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    f1 = f1_score(y_true_valid, y_pred_valid, average="macro")
    precision = precision_score(y_true_valid, y_pred_valid, average="macro")
    recall = recall_score(y_true_valid, y_pred_valid, average="macro")
    
    metrics_summary.append({
        "Variant": variant_name,
        "Valid_Responses": f"{len(valid_indices)}/{len(texts_to_classify)}",
        "Accuracy": round(accuracy * 100, 2),
        "F1_Score": round(f1 * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "Tempo_Execucao": str(tempo_variant)
    })
    
    sns.set_context("paper", font_scale=1.2)
    
    # confusion matrix
    plt.figure(figsize=(6, 5))
    cm_llm = confusion_matrix(y_true_valid, y_pred_valid)
    sns.heatmap(cm_llm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'])
    plt.title(f'Confusion Matrix - Llama 3.1 {variant_name}', pad=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_llama_{variant_name}.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # ROC curve
    fpr_llm, tpr_llm, _ = roc_curve(y_true_valid, y_probs_valid)
    roc_auc_llm = auc(fpr_llm, tpr_llm)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_llm, tpr_llm, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_llm:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - Llama 3.1 {variant_name}', pad=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_curve_llama_{variant_name}.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

# final CSV summary
df_metrics = pd.DataFrame(metrics_summary)
csv_path = os.path.join(output_dir, 'ollama_metrics_summary.csv')
df_metrics.to_csv(csv_path, index=False)

end_total = time.time()
print("\n========================================")
print("Execution completed successfully!")
print(f"Total Time: {datetime.timedelta(seconds=int(end_total - start_total))}")
print(f"The artifacts are in the folder: {output_dir}/")
print(df_metrics.to_string(index=False))