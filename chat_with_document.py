from pathlib import Path
NEON_GREEN = '\033[92m'
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET_COLOR = '\033[0m'

import os

def read_text_file(file_path):
    """Read content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return ""

def split_into_chunks(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks for easier reading."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def main():
    # Read the Satori Protocol text file
    script_dir = Path(__file__).parent
    file_path = script_dir / "source.txt"
    text = read_text_file(file_path)
    
    if not text:
        print("Could not read the text file.")
        return
    
    # Split into manageable chunks
    chunks = split_into_chunks(text)
    
    # Initialize LLM
    llm = ChatOllama(model="gemma3:4b", base_url="http://localhost:11434")

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that answers questions based ONLY on the provided context. If the answer is not in the context, state that you don't have it in the source file so you cant answer."),
        ("user", "Context: {context}\n\nQuestion: {question}")
    ])

    # Create RAG chain
    chain = prompt | llm | StrOutputParser()

    print(NEON_GREEN+"\n-------------LLM TEXT CHAT ------------------"+ RESET_COLOR)
    while True:
        question = input(PINK+"\nAsk your question about the source document (q to quit): "+RESET_COLOR)
        if question.lower() == 'q':
            break
            
        # Simple keyword-based search in chunks to create context
        relevant_chunks = []
        keywords = question.lower().split()
        for chunk in chunks:
            if any(keyword in chunk.lower() for keyword in keywords):
                relevant_chunks.append(chunk)
        
        context = "\n\n".join(relevant_chunks)

        if context:
            print(PINK + "\nGenerating response..."+ RESET_COLOR)
            response = chain.invoke({"context": context, "question": question})
            print(CYAN +f"\nAnswer: {response}"+ RESET_COLOR)
        else:
            print(YELLOW +"\nNo relevant information found in the document for your question. Please try rephrasing."+ RESET_COLOR)

if __name__ == "__main__":
    main()