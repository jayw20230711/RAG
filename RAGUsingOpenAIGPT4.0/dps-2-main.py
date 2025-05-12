import os
import openai
# this is for openai version 1.37.2 ----- dps - 2
import time
from dotenv import load_dotenv
from openai import OpenAI

# Initialize with proper version handling
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("Error: OPENAI_API_KEY is not set in the environment. Please set it in the .env file.")

# client = OpenAI(api_key=openai_api_key)
client = openai.OpenAI(api_key=openai.api_key)

def get_or_create_vector_store(client, name="rag_documents"):
    """Compatible version for older API clients"""
    try:
        # First upload files
        file_ids = []
        for filename in os.listdir("Upload"):
            if filename.endswith(".pdf"):
                with open(f"Upload/{filename}", "rb") as f:
                    file = client.files.create(file=f, purpose="assistants")
                    file_ids.append(file.id)
                    print(f"Uploaded {filename} (ID: {file.id})")

        # Create assistant with file IDs directly
        print("Creating assistant with file references (no vector store in this version)")
        return {"file_ids": file_ids}  # Return file references instead

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def get_or_create_assistant(client, model, name, file_references):
    """Modified assistant creation without vector stores"""
    try:
        assistants = client.beta.assistants.list()
        for asst in assistants.data:
            if asst.name == name:
                print(f"Using existing assistant: {asst.id}")
                return asst

        return client.beta.assistants.create(
            name=name,
            model=model,
            tools=[{"type": "retrieval"}],  # Changed from file_search
            file_ids=file_references["file_ids"] if file_references else []
        )
    except Exception as e:
        print(f"Assistant error: {str(e)}")
        return None


# Main execution
try:
    # Get file references (no vector store)
    file_references = get_or_create_vector_store(client)
    if not file_references:
        raise RuntimeError("Failed to process files")

    # Create assistant
    assistant = get_or_create_assistant(
        client,
        model="gpt-4-turbo",
        name="doc_helper",
        file_references=file_references
    )

    # Rest of your thread handling code...

except Exception as e:
    print(f"Fatal error: {str(e)}")
