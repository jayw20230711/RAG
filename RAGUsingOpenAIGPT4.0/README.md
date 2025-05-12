# RAG using OpenAI's Assistant API : GPT-Series

## Prerequisites

- Python 3.11 or higher
- An OpenAI API Key
- PDF documents to serve as the knowledge base

## Installation
1. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Add your OpenAI API key to the `.env` file in the same directory as `*.py` programs:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Setting Up

1. Place your PDF documents in `Upload` folder in the same directory as `*.py`.
2. The environment variables will be loaded from the `.env` file, and the script will automatically interact with the OpenAI API.

## How to Use

1. **Upload PDFs to Vector Store:**
   The function `upload_PDFs_to_vector_store` allows you to upload your PDF documents to the vector store, making them searchable by the RAG system.

2. **Create a Vector Store:**
   The function `get_or_create_vector_store` either retrieves an existing vector store or creates a new one for storing vector embeddings of documents.

3. **Create an AI Assistant:**
   Use `get_or_create_assistant` to create or retrieve an AI assistant capable of querying the vector store for relevant information.

4. **Start a Conversation:**
   After setting up the assistant, you can start a conversation using the Assistant API. The conversation thread will process user inputs, retrieve relevant information from the vector store, and respond with citations if applicable.

5. **Run the Conversation:**
   The assistant runs in a real-time conversation loop, where you can ask questions, and the assistant retrieves information from the PDFs to provide meaningful responses.

## Example Usage

1. **Create and upload PDFs:**
    ```python
    vectorStore = get_or_create_vector_store(client, "your_vector_store_name_here")
    ```

2. **Create an AI Assistant:**
    ```python
    assistant = get_or_create_assistant(client, model_name, "your_assistant_name_here", vector_store.id)
    ```

3. **Start a conversation:**
    ```python
    message_thread = client.beta.threads.create(**thread_conversation)
    ```

## Best Practices

- Ensure PDFs are well-structured and relevant to the domain for improved performance.
- Use meaningful filenames and metadata for better file management.
- Tune parameters like `temperature` and `top_p` for the assistant based on your specific use case.
- Utilize query augmentation and context to improve the retrieval quality in RAG systems.


