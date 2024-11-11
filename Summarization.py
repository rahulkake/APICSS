# Install necessary packages

from huggingface_hub import InferenceClient
from transformers import pipeline
import model_metrics
import mlflow
import docx2txt
import time

# Set the MLflow tracking URI to 'http'
mlflow.set_tracking_uri("http://localhost:5000")

def extract_text_from_docx(docx_path):
    """
    Extracts text from a Word document.

    Parameters:
    docx_path (str): Path to the Word document.

    Returns:
    str: Extracted text from the document.
    """
    return docx2txt.process(docx_path)

def summarize_document(context):
    with mlflow.start_run():
        start_time = time.time()  # Record start time
        summarizer =  pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(context, max_length=750, min_length=128, do_sample=False)
        result = summary[0]['summary_text']
        # log metrics
        model_metrics.log_metrics("summarization", result, start_time, "facebook/bart-large-cnn")
        return result

if __name__ == "__main__":
    docx_path = './datastore/IndvsSL.docx'
    context = extract_text_from_docx(docx_path)
    print(summarize_document(context))
