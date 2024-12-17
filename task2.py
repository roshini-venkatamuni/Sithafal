import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Scrape website content
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract text content (you can customize this to extract more specific data)
    paragraphs = soup.find_all('p')
    text_chunks = [para.get_text() for para in paragraphs if para.get_text()]
    return text_chunks

# Step 2: Convert text to TF-IDF embeddings
def get_tfidf_embeddings(text_chunks):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_chunks)
    return tfidf_matrix, vectorizer

# Step 3: Convert query to TF-IDF embedding
def query_to_embedding(query, vectorizer):
    query_tfidf = vectorizer.transform([query])
    return query_tfidf

# Step 4: Perform similarity search
def search_similar_chunks(query_embedding, tfidf_matrix):
    similarities = cosine_similarity(query_embedding, tfidf_matrix)
    return similarities[0]

# Step 5: Generate response
def generate_response(similarities, text_chunks):
    # Find the most similar chunk
    most_similar_index = similarities.argmax()
    return text_chunks[most_similar_index]

# Example usage
# url = 'https://github.com/anushachappidi/RockPaperScissorsGame'  # Replace with the URL of your target website
url = 'https://github.com/anushachappidi/RockPaperScissorsGame'
query = "What is the main topic of the article?"

# Step 1: Scrape the website
text_chunks = scrape_website(url)

# Step 2: Convert the text chunks to TF-IDF embeddings
tfidf_matrix, vectorizer = get_tfidf_embeddings(text_chunks)

# Step 3: Convert the user's query to a TF-IDF embedding
query_embedding = query_to_embedding(query, vectorizer)

# Step 4: Perform similarity search to find the most relevant chunk
similarities = search_similar_chunks(query_embedding, tfidf_matrix)

# Step 5: Generate a response based on the most similar chunk
response = generate_response(similarities, text_chunks)

print("Response:", response)
