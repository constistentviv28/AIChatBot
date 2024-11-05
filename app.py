import streamlit as st
import PyPDF2
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the LLM
qa_pipeline = pipeline("question-answering")
embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text.strip()

# Function to handle PDF upload
def handle_pdf_upload():
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        return extract_text_from_pdf("uploaded.pdf")
    return None

# Function to create embeddings for the PDF text
def create_embeddings(text):
    sentences = text.split('. ')
    embeddings = embed_model.encode(sentences)
    return sentences, embeddings

# Main function for Streamlit UI
def main():
    st.title("AI Chat with PDF Knowledge Base")

    # Upload PDF and extract text
    pdf_text = handle_pdf_upload()

    if pdf_text:
        st.success("PDF uploaded successfully!")
        st.text_area("Extracted Text", pdf_text, height=200)

        # Create embeddings for the PDF text
        sentences, embeddings = create_embeddings(pdf_text)
        embedding_index = faiss.IndexFlatL2(embeddings.shape[1])
        embedding_index.add(np.array(embeddings))

        # User question input
        user_question = st.text_input("Ask a question about the PDF:")

        if user_question:
            # Embed the user's question
            question_embedding = embed_model.encode([user_question])
            distances, indices = embedding_index.search(question_embedding, k=5)

            # Retrieve relevant sentences
            relevant_contexts = [sentences[i] for i in indices[0]]
            combined_context = " ".join(relevant_contexts)

            # Generate response using the LLM
            response = qa_pipeline(question=user_question, context=combined_context)
            st.write("**AI Response:**", response['answer'])

if __name__ == "__main__":
    main()
