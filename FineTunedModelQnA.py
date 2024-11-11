import time
import model_metrics
import mlflow
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(api_key="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")
ft_model = "ft:gpt-4o-mini-2024-07-18:personal::AOjr3vRY" # created once.

def create_ft_model():
    # Step 1: Upload the dataset in binary mode
    with open("./datastore/IndvsSL.jsonl", "rb") as f:  # Open file in binary mode
        response = client.files.create(
            file=f,
            purpose='fine-tune'
        )

    print(response)
    # Access the file ID correctly
    file_id = response.id  # Use dot notation to access the id attribute
    print("Uploaded file ID:", file_id)

    # Step 2: Create a fine-tuning job
    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",  # Choose the base model
    )
    print("Fine-tuning job created:", fine_tune_response)
    job_id = fine_tune_response.id


    # Step 3: Monitor the fine-tuning job until it's complete
    while True:
        status_response = client.fine_tuning.jobs.retrieve(job_id)
        print("Current status:", status_response.status)
        print(status_response)

        if status_response.status in ['succeeded', 'failed']:
            break
        
        time.sleep(60)  # Wait a minute before checking again

    # Step 4: Use the fine-tuned model for inference
    print(status_response.fine_tuned_model)
    ft_model = status_response.fine_tuned_model

def ask_question_with_ft(question):
    with mlflow.start_run():
        start_time = time.time()  # Record start time
        response = client.chat.completions.create(
            model=ft_model,
            #prompt="Who captained the Sri Lankan team in the final?\n",
            messages=[
                {"role": "system", "content": "Cricket Guru us a factual cricket analytics assistant."},
                {"role": "user", "content": question}
            ],
            max_tokens=50
        )
        result = response.choices[0].message.content

        # log metrics
        model_metrics.log_metrics(question, result, start_time, ft_model)
        print("Model response:", result)

def chatbot_with_ft():
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        answer = ask_question_with_ft(user_input)
    
if __name__ == "__main__":
    #create_ft_model()
    chatbot_with_ft()
