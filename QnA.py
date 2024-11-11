# Install necessary packages

from huggingface_hub import InferenceClient
import model_metrics
import mlflow
import time
import docx2txt

# Set the MLflow tracking URI to 'http'
mlflow.set_tracking_uri("http://localhost:5000")

# Initialize Hugging Face Inßerence Client
client = InferenceClient(api_key="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")

def extract_text_from_docx(docx_path):
    """
    Extracts text from a Word document.

    Parameters:
    docx_path (str): Path to the Word document.

    Returns:
    str: Extracted text from the document.
    """
    return docx2txt.process(docx_path)

def ask_question_with_phi(question, context):
    """
    Asks a question using the 'microsoft/Phi-3.5-mini-instruct' model with optional context.

    Parameters:
    question (str): The question to ask.
    context (str, optional): Additional context from the document.

    Returns:
    str: The model's response.
    """
    with mlflow.start_run():  # Start an MLFlow run
        try:
            start_time = time.time()  # Record start time
            messages = [
                { "role": "user", "content": f"{context}\n\nQuestion: {question}" }
            ]

            output = client.chat.completions.create(
                model="microsoft/Phi-3.5-mini-instruct",
                messages=messages,
                stream=True,
                temperature=0.5,
                max_tokens=1024,
                top_p=0.7
            )

            # Collect all chunks in a list and join them after the loop
            full_response = []
            for chunk in output:
                full_response.append(chunk.choices[0].delta.content)
            result = "".join(full_response)
            # log metrics
            model_metrics.log_metrics(question, result, start_time, "microsoft/Phi-3.5-mini-instruct")
            return result

        except Exception as e:
            return f"Error: {str(e)}"

def chatbot_with_document(context):
    """
    Simple chatbot that uses 'microsoft/Phi-3.5-mini-instruct' to answer based on the document content.

    Parameters:
    context (str): The context for answering questions.
    """
    print("Hello! I am your chatbot. Ask me anything, or type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        answer = ask_question_with_phi(user_input, context)
        print("Bot:", answer)

if __name__ == "__main__":
    docx_path = './datastore/IndvsSL.docx'
    context = extract_text_from_docx(docx_path)
    chatbot_with_document(context)
