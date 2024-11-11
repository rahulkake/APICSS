import docx2txt
import model_metrics
import mlflow
import time

from gtts import gTTS
from openai import OpenAI
from transformers import pipeline
from huggingface_hub import InferenceClient

client = OpenAI(api_key="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")

def image_generation(context):
    summary = summarize_document(context)

    with mlflow.start_run():
        start_time = time.time()  # Record start time
        response = client.images.generate(
            prompt=summary[:64],
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url

        # log metrics
        model_metrics.log_metrics("image-generation", image_url, start_time, "openAI/DALL-E")
        print(image_url)

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
    summarizer =  pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(context, max_length=100, min_length=64, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    docx_path = './datastore/IndvsSL.docx'
    context = extract_text_from_docx(docx_path)
    image_generation(context)