import sys
import time
import json
import os
import docx2txt
import argparse
import QnA, Speech, Summarization, Translation, ImageGeneration, FineTunedModelQnA
from huggingface_hub import InferenceClient


def extract_text_from_docx(docx_path):
    """
    Extracts text from a Word document.

    Parameters:
    docx_path (str): Path to the Word document.

    Returns:
    str: Extracted text from the document.
    """
    return docx2txt.process(docx_path)

def main():
    # Define available tasks
    tasks = ['QnA', 'Summarisation', 'TranslationToHindi', 'ListenToSummary', 'GenerateAnImage', 'FineTunedModel']
    
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Choose an AI task and provide the path to the document to be processed."
    )
    
    # Define the task argument
    parser.add_argument(
        "-t", "--task",
        type=str,
        choices=tasks,
        required=True,
        help="Specify the AI task to run. Options: 'QnA', 'Summarisation', 'TranslationToHindi', 'ListenToSummary', 'GenerateAnImage', 'FineTunedModel'"
    )
    
    # Define the document path argument
    parser.add_argument(
        "-d", "--document",
        type=str,
        help="Specify the path to the document file."
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    #  document required only if the task is not FineTunedModel
    if args.task != 'FineTunedModel' and args.document is None:
        parser.error("The --document argument is required for the selected task.")

    if args.task != 'FineTunedModel':
        context = extract_text_from_docx(args.document)

    # Output task and document path
    print(f"Task selected: {args.task}")
    print(f"Document path provided: {args.document}")
    
    # Logic to handle different tasks can be added here
    if args.task == "QnA":
        print("Running Question and Answer task...")
        QnA.chatbot_with_document(context)
    elif args.task == "Summarisation":
        print("Running Summarisation task...")
        print(Summarization.summarize_document(context))
    elif args.task == "TranslationToHindi":
        print("Running Translation to Hindi task...")
        print(Translation.translate(context))
    elif args.task == "ListenToSummary":
        print("Running Speech Synthesis task...")
        Speech.tts(context)
    elif args.task == "GenerateAnImage":
        print("Runing Image Generation...")
        ImageGeneration.image_generation(context)
    elif args.task == "FineTunedModel":
        print("Runing FineTunedModel...")
        FineTunedModelQnA.chatbot_with_ft()
    else:
        print("Unknown task selected.")

if __name__ == "__main__":
    main()

