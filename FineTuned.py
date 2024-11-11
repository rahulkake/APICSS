import pandas as pd
from transformers import Trainer, TrainingArguments, DistilBertForQuestionAnswering, DistilBertTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import pipeline

# Creating a synthetic dataset with answers
data = {
    'question': [
        "What is the capital of France?",
        "Who wrote 'Hamlet'?",
        "What is the boiling point of water?",
        "What is the largest planet in our solar system?",
        "What language is spoken in Brazil?",
        "Who painted the Mona Lisa?",
        "What is the speed of light?",
        "Who is the author of '1984'?",
        "What is the chemical formula for water?",
        "What is the smallest country in the world?"
    ],
    'context': [
        "The capital of France is Paris.",
        "'Hamlet' is a tragedy written by William Shakespeare.",
        "The boiling point of water is 100 degrees Celsius.",
        "Jupiter is the largest planet in our solar system.",
        "Portuguese is the language spoken in Brazil.",
        "The Mona Lisa was painted by Leonardo da Vinci.",
        "The speed of light is approximately 299,792 kilometers per second.",
        "'1984' is a dystopian novel by George Orwell.",
        "The chemical formula for water is H2O.",
        "The smallest country in the world is Vatican City."
    ],
    'answers': [
        "Paris",  # Answer for the first question
        "William Shakespeare",  # Answer for the second question
        "100 degrees Celsius",  # Answer for the third question
        "Jupiter",  # Answer for the fourth question
        "Portuguese",  # Answer for the fifth question
        "Leonardo da Vinci",  # Answer for the sixth question
        "299,792 kilometers per second",  # Answer for the seventh question
        "George Orwell",  # Answer for the eighth question
        "H2O",  # Answer for the ninth question
        "Vatican City"  # Answer for the tenth question
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
pretrained_model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Function to find answer positions
def find_answer_positions(context, answer):
    start = context.find(answer)
    end = start + len(answer) - 1
    return start, end

# Add start and end positions to the DataFrame
df['start_positions'], df['end_positions'] = zip(*df.apply(lambda row: find_answer_positions(row['context'], row['answers']), axis=1))

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenization function with padding
def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding='max_length',  # Pad to the maximum length
        max_length=512  # Adjust this value based on your model's requirements
    )
    # Set the start and end positions
    tokenized_inputs['start_positions'] = examples['start_positions']
    tokenized_inputs['end_positions'] = examples['end_positions']
    return tokenized_inputs

# Tokenizing the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=pretrained_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the fine-tuned model and tokenizer
fine_tuned_model_path = "./qa_model"
pretrained_model.save_pretrained(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)

# Load the fine-tuned model for inference
fine_tuned_qa_pipeline = pipeline("question-answering", model=fine_tuned_model_path, tokenizer=fine_tuned_model_path)

# Load the pre-trained model for inference
pretrained_qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

# Custom question and context
context = "The capital of France is Paris."
question = "What is the capital of France?"

# Get the answer from the pre-trained model
pretrained_result = pretrained_qa_pipeline(question=question, context=context)

# Get the answer from the fine-tuned model
fine_tuned_result = fine_tuned_qa_pipeline(question=question, context=context)

# Print the results
print(f"Pre-trained Model Answer: {pretrained_result['answer']}")
print(f"Fine-tuned Model Answer: {fine_tuned_result['answer']}")
