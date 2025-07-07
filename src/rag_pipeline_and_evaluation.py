
from google.colab import drive
drive.mount('/content/drive')

import faiss
import pickle
import pandas as pd

# Load FAISS index
index = faiss.read_index('/content/drive/My Drive/vector_store/faiss_index.index')

# Load metadata (if needed)
with open('/content/drive/My Drive/vector_store/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Load the original or cleaned complaint chunks CSV
df = pd.read_csv('/content/drive/My Drive/filtered_complaints.csv')

from sentence_transformers import SentenceTransformer
from transformers import pipeline

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text2text-generation", model="google/flan-t5-base")

import numpy as np

def retrieve_top_k_chunks(question, k=5):
    query_embedding = embedding_model.encode([question])
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    for i in indices[0]:
        if i >= 0 and i < len(df):
            retrieved_chunks.append(df.iloc[i]['cleaned_narrative'])  # Use 'cleaned_narrative' here
    return retrieved_chunks




def build_prompt(context_chunks, question, max_context_tokens=400):
    # Simple token length estimation by word count (can use a tokenizer for better accuracy)
    tokens_so_far = 0
    truncated_chunks = []
    for chunk in context_chunks:
        chunk_tokens = len(chunk.split())
        if tokens_so_far + chunk_tokens > max_context_tokens:
            # Truncate chunk to fit remaining tokens
            words = chunk.split()
            allowed = max_context_tokens - tokens_so_far
            truncated_chunks.append(" ".join(words[:allowed]))
            break
        else:
            truncated_chunks.append(chunk)
            tokens_so_far += chunk_tokens

    context = "\n\n".join(truncated_chunks)
    prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""
    return prompt


def answer_question_rag(question):
    chunks = retrieve_top_k_chunks(question, k=5)
    prompt = build_prompt(chunks, question)
    response = generator(prompt, max_new_tokens=256, do_sample=False)[0]['generated_text']
    return {
        "question": question,
        "answer": response.strip(),
        "retrieved_sources": chunks[:2]
    }

print(df.columns.tolist())

eval_questions = [
    "What issues do customers face with Buy Now Pay Later?",
    "How do consumers describe credit card disputes?",
    "Are there any recurring complaints about personal loans?",
    "Do customers have trouble with money transfers?",
    "What problems are reported with savings accounts?",
    "Are there complaints about account closures?",
    "Do customers report unauthorized charges?",
    "How is customer service described in complaints?",
    "Do customers complain about overdraft fees?",
    "What are common issues with loan servicing?"
]

results = []

for i, q in enumerate(eval_questions):
    print(f"Processing Q{i+1}: {q}")
    result = answer_question_rag(q)
    result["quality_score"] = ""  # To be filled manually
    result["comments"] = ""       # To be filled manually
    results.append(result)

eval_df = pd.DataFrame(results)
eval_df.to_csv("/content/drive/My Drive/evaluation_results.csv", index=False)

# Preview
eval_df.head()