import pandas as pd
from datasets import load_dataset, load_metric, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import sys
from tqdm import tqdm # Import tqdm for a progress bar

# ----------------------------------------------------------------------
# 1. SETUP: MODEL AND DATA LOADING
# ----------------------------------------------------------------------

print("--- 1. Initializing Evaluation Setup ---")

# Define the path where the pre-processed dataset is stored
dataset_path = '/DATA/pranta_2411ai09/DialogueSummarization/data/samsum_dataset'

# Load the dataset (which should contain the 'test' split)
try:
    dataset_samsum = load_from_disk(dataset_path)
    print(f"Dataset loaded successfully from: {dataset_path}")
except Exception as e:
    print(f"ERROR: Could not load dataset from {dataset_path}. Check the path.")
    print(f"Details: {e}")
    sys.exit(1)

# Define the path where the fine-tuned model checkpoint is saved
model_checkpoint_path = './pegasus-samsum'

# Load the fine-tuned model and tokenizer
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
    print(f"Fine-tuned model and tokenizer loaded from: {model_checkpoint_path}")
except Exception as e:
    print(f"ERROR: Could not load model from checkpoint. Check the path.")
    print(f"Details: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------
# 2. BATCH PROCESSING AND GENERATION
# ----------------------------------------------------------------------

batch_size = 8
test_dataset = dataset_samsum['test']
num_samples = len(test_dataset)

# Initialize lists to store the results
generated_summaries = []
reference_summaries = []

print(f"\n--- 2. Starting Generation on Test Set ({num_samples} samples) ---")
print(f"Processing in batches of size: {batch_size}")

# Loop through the test dataset in batches
# Using tqdm for a clear progress bar in the terminal
for start_idx in tqdm(range(0, num_samples, batch_size), desc="Generating Summaries"):
    end_idx = min(start_idx + batch_size, num_samples)
    
    # Extract the dialogues for the current batch
    batch_dialogues = test_dataset[start_idx:end_idx]['dialogue']
    
    # Tokenize the dialogues, converting them to PyTorch tensors
    inputs = tokenizer(
        batch_dialogues, 
        return_tensors='pt', 
        truncation=True, 
        padding='longest', 
        max_length=1024
    )
    
    # Move tokenized inputs to the same device as the model (GPU/CPU)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate summary IDs using the model
    with torch.no_grad(): # Disable gradient calculation for inference
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_beams=4,        # Beam search width (explores 4 sequences)
            length_penalty=1.4, # Encourages longer sequences
            max_length=100,     # Maximum length of generated summary
            min_length=25,      # Minimum length of generated summary
            early_stopping=True # Stop beam search early if minimum scores are reached
        )
    
    # Decode the generated token IDs back into human-readable strings
    batch_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    
    # Store the results
    generated_summaries.extend(batch_summaries)
    reference_summaries.extend(test_dataset[start_idx:end_idx]['summary'])

print("\nGeneration complete. Total summaries generated.")

# ----------------------------------------------------------------------
# 3. ROUGE SCORE CALCULATION
# ----------------------------------------------------------------------

print("--- 3. Computing ROUGE Metrics ---")

# Load the ROUGE metric calculator
rouge = load_metric('rouge')

# Compute ROUGE scores comparing generated summaries to reference summaries
results = rouge.compute(
    predictions=generated_summaries, 
    references=reference_summaries, 
    use_stemmer=True # Optional: use stemming for minor linguistic variations
)

# Extract the F1 (mid.fmeasure) scores for standard ROUGE metrics
rouge_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
rouge_dict = {rn: results[rn].mid.fmeasure for rn in rouge_names}

# ----------------------------------------------------------------------
# 4. DISPLAY RESULTS
# ----------------------------------------------------------------------

print("\n--- 4. Final Evaluation Results (F1-Scores) ---")

# Convert the results dictionary into a Pandas DataFrame for clear tabular display
results_df = pd.DataFrame(rouge_dict, index=['pegasus'])

# Print the DataFrame
print(results_df)

print("\nEvaluation Script Finished.")
