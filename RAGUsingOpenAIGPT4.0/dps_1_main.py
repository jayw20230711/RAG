import os
import openai
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")

client = openai.OpenAI(api_key=openai_api_key)
model_name = "gpt-4o"


def upload_pdfs_to_vector_store(client, vector_store_id, directory_path):
    """Uploads PDFs to a vector store using current API methods."""
    try:
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory '{directory_path}' not found")

        file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
                      if f.lower().endswith('.pdf')]

        if not file_paths:
            raise ValueError("No PDF files found in directory")

        file_ids = []
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                # First upload the file
                file = client.files.create(file=f, purpose="assistants")
                print(f"Uploaded {os.path.basename(file_path)} (ID: {file.id})")
                file_ids.append(file.id)

        # Then add to vector store
        if file_ids:
            client.beta.vector_stores.files.create_and_poll(
                vector_store_id=vector_store_id,
                file_ids=file_ids
            )
        return True

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return False


def get_or_create_vector_store(client, name):
    """Gets or creates a vector store with the given name."""
    try:
        # Check existing stores
        stores = client.beta.vector_stores.list()
        for store in stores.data:
            if store.name == name:
                print(f"Using existing vector store: {store.id}")
                return store

        # Create new store
        print(f"Creating new vector store: {name}")
        store = client.beta.vector_stores.create(name=name)

        # Upload files
        if not upload_pdfs_to_vector_store(client, store.id, 'Upload'):
            raise RuntimeError("Failed to upload files")

        return store

    except Exception as e:
        print(f"Vector store error: {str(e)}")
        return None


def get_or_create_assistant(client, model, name, vs_id):
    """Gets or creates an assistant with the given configuration."""
    try:
        assistants = client.beta.assistants.list()
        for asst in assistants.data:
            if asst.name == name:
                print(f"Using existing assistant: {asst.id}")
                return asst

        print(f"Creating new assistant: {name}")
        return client.beta.assistants.create(
            name=name,
            model=model,
            instructions="You are a helpful technical assistant...",
            tools=[{"type": "file_search"}],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vs_id]
                }
            }
        )
    except Exception as e:
        print(f"Assistant error: {str(e)}")
        return None


# Main execution
try:
    # Setup vector store
    vector_store = get_or_create_vector_store(client, "rag_llm_vector_store")
    if not vector_store:
        raise RuntimeError("Failed to setup vector store")

    # Setup assistant
    assistant = get_or_create_assistant(client, model_name, "mygpt_assistant", vector_store.id)
    if not assistant:
        raise RuntimeError("Failed to setup assistant")

    # Create thread
    thread = client.beta.threads.create()
    print(f"Thread created: {thread.id}")

    # Chat loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ('exit', 'quit'):
            break

        # Add message
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input
        )

        # Run assistant
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread.id)
            for msg in messages:
                if msg.role == "assistant":
                    for content in msg.content:
                        if content.type == "text":
                            print(f"\nAssistant: {content.text.value}")
        else:
            print(f"Run failed with status: {run.status}")

except Exception as e:
    print(f"Fatal error: {str(e)}")