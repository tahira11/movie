import os
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the Groq client
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # Make sure GROQ_API_KEY is set in your environment variables

# Load the dataset (ensure your dataset path is correct)
dataset_path = "/content/movie_dataset.csv"
movies_df = pd.read_csv(dataset_path)

# Preprocess the dataset by creating summaries and vectors
def preprocess_data(df):
    # Combine relevant text columns to form a concise summary for each movie
    df['summary'] = df.apply(lambda row: f"{row['title']} ({row['release_date']}): {row['overview']} "
                                         f"Genres: {row['genres']} Keywords: {row['keywords']}", axis=1)
    return df

movies_df = preprocess_data(movies_df)

# Convert summaries to TF-IDF vectors for retrieval
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies_df['summary'])

# Define function to retrieve similar movies based on a query
def retrieve_similar_movies(query, df, tfidf_matrix, top_n=5):
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# Call Groq API for generation based on the retrieved summaries and query
def generate_summary_with_groq(query, retrieved_text):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"{query}\n\nRelated information:\n{retrieved_text}"}
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Main interactive loop
def rag_application():
    print("Welcome to the Movie RAG-based Application!")
    while True:
        # Prompt user for a query
        user_query = input("Ask a question about movies or type 'exit' to quit: ")

        if user_query.lower() in ['exit', 'no', 'quit']:
            print("Exiting the application. Goodbye!")
            break

        # Retrieve relevant movie summaries
        retrieved_movies = retrieve_similar_movies(user_query, movies_df, tfidf_matrix)
        retrieved_summaries = " ".join(retrieved_movies['summary'].values)

        # Generate a summary response based on retrieved movies
        generated_summary = generate_summary_with_groq(user_query, retrieved_summaries)
        print("Generated Summary:", generated_summary)

        # Ask if user wants to continue or exit
        continue_query = input("Do you have another question? (yes/no): ")
        if continue_query.lower() != 'yes':
            print("Exiting the application. Goodbye!")
            break

# Run the application
rag_application()
