# app.py

import faiss
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import gradio as gr

# Load components
index = faiss.read_index('vector_store/faiss_index.index')

with open('vector_store/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

df = pd.read_csv('filtered_complaints.csv')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Core RAG logic
def retrieve_top_k_chunks(question, k=5):
    query_embedding = embedding_model.encode([question])
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [df.iloc[i]['cleaned_narrative'] for i in indices[0] if i >= 0 and i < len(df)]
    return retrieved_chunks

def build_prompt(context_chunks, question, max_context_tokens=400):
    tokens_so_far = 0
    truncated_chunks = []
    for chunk in context_chunks:
        chunk_tokens = len(chunk.split())
        if tokens_so_far + chunk_tokens > max_context_tokens:
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
        "answer": response.strip(),
        "retrieved_sources": chunks[:2]
    }

def gradio_interface(question):
    result = answer_question_rag(question)
    sources = "\n\n".join(result["retrieved_sources"])
    return result["answer"], sources

# UI
with gr.Blocks() as demo:
    gr.Markdown("# CrediTrust RAG Chatbot")
    gr.Markdown("Ask a question about customer complaints.")

    with gr.Row():
        question_input = gr.Textbox(label="Enter your question")
        submit_btn = gr.Button("Submit")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer", lines=4)
        sources_output = gr.Textbox(label="Retrieved Sources", lines=8)

    with gr.Row():
        clear_btn = gr.Button("Clear")

    submit_btn.click(fn=gradio_interface, inputs=question_input, outputs=[answer_output, sources_output])
    clear_btn.click(fn=lambda: ("", ""), inputs=[], outputs=[answer_output, sources_output])

demo.launch()
