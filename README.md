
# Smart-PDF-QA-Bot

## Features
- PDF document ingestion and processing
- Vector store creation for efficient information retrieval
- Real-time streaming responses
- Persistent chat history
- User-friendly Streamlit interface

## Requirements
- Python 3.7+
- Streamlit
- LangChain
- Faiss
- PyPDF
- Ollama

## Installation
1. Clone the repository
2. Create and activate a virtual environment (recommended)
3. Install required packages:
   ```sh
   pip install -r requirements.txt
   ```
4. Ensure Ollama is installed and running with Llama2. Start the Ollama service before starting the app.

## Configuration
1. Place your PDF in the project directory
2. Update `PDF_PATH` in `app.py`
3. Adjust Ollama settings if necessary

## Usage
1. Run: `streamlit run app.py`
2. Open the provided URL in your browser
3. Wait for initialization
4. Start asking questions about your PDF

## How It Works
1. **PDF Processing**
2. **Text Splitting**
3. **Embedding Creation**
4. **Vector Store**
5. **Query Processing**
6. **Response Generation**
7. **Streaming Output**

## Project Structure
- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `corpus_vectorstore/`: Processed vector store (created on first run)

## Customization
- Change language model by updating `OLLAMA_MODEL`
- Adjust text splitting parameters
- Modify prompt template

## Troubleshooting
- Ensure PDF is readable and not encrypted
- Check Ollama service is running
- For slow initialization, consider pre-processing

## Contributing
Contributions welcome. Submit a Pull Request.

## License
Open-source under the MIT License
