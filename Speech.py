
import docx2txt
import model_metrics
import mlflow
import time

from gtts import gTTS
from openai import OpenAI
from transformers import pipeline
from huggingface_hub import InferenceClient

def tts(context):
    client = OpenAI(api_key="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")

    summary = summarize_document(context)
    # Convert the text to speech
    tts = gTTS(text=summary, lang='en')

    # Save the audio file
    tts.save("output_audio.mp3")

    with mlflow.start_run():
        start_time = time.time()  # Record start time
        # Transcribe Audio with Whisper API
        with open("output_audio.mp3", "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        # log metrics
        model_metrics.log_metrics("speech-synthesis", response.text, start_time, "openAI/whisper-1")

        print("Transcribed Text:", response.text)
        print("")
        print("Audio File is generated and saved as: output_audio.mp3 in the current directory")

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
    summary = summarizer(context, max_length=750, min_length=128, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    docx_path = './datastore/IndvsSL.docx'
    context = extract_text_from_docx(docx_path)
    print(tts(context))