import streamlit as st
import os
from langchain.vectorstores import Chroma
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Function to create a question-answering chain
def create_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa_chain

# Main Streamlit app
def main():
    st.title("Document QA App")

    # Set the persist_directory here
    persist_directory = 'db_pro'

    # Load the pre-embedded documents
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=None)
    qa_chain = create_qa_chain(vectordb)

    # User input for questions
    question = st.text_input("Ask a question about your documents:")
    
    # Check if a question is provided
    if question:
        if st.button("Get Answer"):
            # Use the question-answering chain to get the answer
            llm_response = qa_chain(question)
            
            # Display the answer
            st.write("Answer:")
            st.write(llm_response['result'])

if __name__ == "__main__":
    main()


