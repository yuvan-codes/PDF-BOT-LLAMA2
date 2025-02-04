import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# Constants
PDF_PATH = "Corpus.pdf"
PERSIST_PREFIX = "corpus_faiss_part_"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"

# Custom callback handler to stream output to Streamlit
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

    def clear(self):
        self.text = ""

# Initialize Streamlit session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.chat_history = []

# Create multiple vectorstores
@st.cache_resource
def create_multiple_vectorstores():
    embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    vectorstores = {}
    num_pages = len(documents)
    pages_per_vectorstore = 5  # Adjust based on relevance
    
    for i in range(0, num_pages, pages_per_vectorstore):
        subset_docs = documents[i : i + pages_per_vectorstore]
        splits = text_splitter.split_documents(subset_docs)
        
        vectorstore_name = f"{PERSIST_PREFIX}{i//pages_per_vectorstore}"
        vector_path = os.path.join(os.getcwd(), vectorstore_name)
        
        print(f"Creating FAISS vectorstore at {vector_path}...")  # Debugging line
        vectorstore = FAISS.from_documents(splits, embedding=embeddings)
        vectorstore.save_local(vector_path)
        
        if not os.path.exists(vector_path):  # Check if saving was successful
            raise ValueError(f"FAISS index saving failed for {vector_path}")
        
        vectorstores[vectorstore_name] = vectorstore

    return vectorstores

# Load vectorstores from saved directories
@st.cache_resource
def load_vectorstores():
    embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    vectorstore_dirs = [d for d in os.listdir() if d.startswith(PERSIST_PREFIX)]
    
    if not vectorstore_dirs:
        raise ValueError("No FAISS vectorstore directories found. Please rerun vectorstore creation.")

    print("Found FAISS directories:", vectorstore_dirs)  # Debugging line
    
    vectorstores = {}
    for directory in vectorstore_dirs:
        vector_path = os.path.join(os.getcwd(), directory)
        print(f"Loading FAISS index from {vector_path}...")  # Debugging line
        vectorstores[directory] = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)

    
    return vectorstores


# Identify the best vectorstore based on the query
def find_best_vectorstore(query, vectorstores):
    best_store = None
    best_score = float("-inf")

    for name, vectorstore in vectorstores.items():
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(query)
        
        if docs:
            relevance_score = sum(len(doc.page_content) for doc in docs)  # Simple heuristic
            if relevance_score > best_score:
                best_score = relevance_score
                best_store = name
    
    return best_store

# Initialize prompt and memory
@st.cache_resource
def initialize_components():
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""You are a knowledgeable chatbot, here to help with questions about the provided document. Your tone should be professional and informative. Provide concise answers.

        Context: {context}
        History: {history}

        User: {question}
        Chatbot:"""
    )
    
    memory = ConversationBufferMemory(memory_key="history", return_messages=True, input_key="question")
    
    return prompt, memory

# Main app
def main():
    st.title("Corpus PDF Chatbot")

    if not st.session_state.initialized:
        vectorstores = create_multiple_vectorstores()
        prompt, memory = initialize_components()
        st.session_state.vectorstores = vectorstores
        st.session_state.prompt = prompt
        st.session_state.memory = memory
        st.session_state.initialized = True
        st.text("Chatbot initialized and ready!")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("You:"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_container = st.empty()
            stream_handler = StreamHandler(response_container)

            # Load vectorstores
            vectorstores = load_vectorstores()

            # Identify the best vectorstore
            best_vectorstore_name = find_best_vectorstore(user_input, vectorstores)
            best_vectorstore = vectorstores[best_vectorstore_name] if best_vectorstore_name else None
            
            if best_vectorstore:
                llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, callbacks=[stream_handler])

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type='stuff',
                    retriever=best_vectorstore.as_retriever(),
                    chain_type_kwargs={
                        "verbose": True,
                        "prompt": st.session_state.prompt,
                        "memory": st.session_state.memory,
                    }
                )

                response = qa_chain(user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": response['result']})
                stream_handler.clear()
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "I couldn't find relevant information in the document."})

if __name__ == "__main__":
    main()
