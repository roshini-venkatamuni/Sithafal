import pdfplumber

# Function to extract text from specific pages
def extract_text_from_pdf(pdf_path, pages):
    extracted_text = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in pages:
            extracted_text[page_num] = pdf.pages[page_num].extract_text()
    return extracted_text

# Example: Extract data from pages 2 and 6 (index starts at 0)
pdf_path = r"C:\Users\admin\Desktop\apple.pdf"
pages_to_extract = [1, 9]  # Page 2 and Page 6
extracted_text = extract_text_from_pdf(pdf_path, pages_to_extract)

print("Page 2 Content:\n", extracted_text[1])
print("Page 6 Content:\n", extracted_text[9])

# 2

from sentence_transformers import SentenceTransformer

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Chunk text into manageable parts
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Example: Chunking and embedding for Page 2
chunks_page2 = chunk_text(extracted_text[1])
chunks_page6 = chunk_text(extracted_text[9])

# Generate embeddings for chunks
embeddings_page2 = embedding_model.encode(chunks_page2)
embeddings_page6 = embedding_model.encode(chunks_page6)

#3

import faiss
import numpy as np

print("Embeddings for Page 2:", embeddings_page2)
print("Shape of Embeddings for Page 2:", embeddings_page2.shape)
print("Embeddings for Page 6:", embeddings_page6)
print("Shape of Embeddings for Page 6:", embeddings_page6.shape)


# Create a FAISS index
dimension = embeddings_page2.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(np.array(embeddings_page2))  # Add Page 2 embeddings
index.add(np.array(embeddings_page6))  # Add Page 6 embeddings

print(f"Total chunks in FAISS index: {index.ntotal}")

#4

# Function to retrieve most relevant chunks for a query
def retrieve_relevant_chunks(query, embedding_model, index, chunks, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=top_k)
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks

# Example Query
query = "What is the unemployment rate for a Bachelor's degree?"
relevant_chunks = retrieve_relevant_chunks(query, embedding_model, index, chunks_page2)

print("Relevant Chunks:\n", relevant_chunks)

#5

import openai

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Generate response from LLM
def generate_response_with_llm(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Based on the following information:\n{context}\n\nAnswer the question: {query}"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

# Generate response for the query
response = generate_response_with_llm(query, relevant_chunks)
print("Generated Response:\n", response)

#6

# Example Comparison Query
comparison_query = "Compare unemployment rates for Bachelor's and Associate degrees."
comparison_chunks = retrieve_relevant_chunks(comparison_query, embedding_model, index, chunks_page2)

# Generate response for comparison
comparison_response = generate_response_with_llm(comparison_query, comparison_chunks)
print("Comparison Response:\n", comparison_response)