# ----------------------------------------------------------------------
# 1. SETUP: IMPORT LIBRARIES AND LOAD PRE-TRAINED MODEL
# ----------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer

# Define the model name for the Pegasus model and its tokenizer
model_name = "google/pegasus-cnn_dailymail"
print(f"--- 1. Initializing Model Setup ---")
print(f"Loading Tokenizer and Model: {model_name}")

# Initialize Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print(f"Model and Tokenizer loaded successfully.")

# ----------------------------------------------------------------------
# 2. DATA SETUP
# ----------------------------------------------------------------------
dataset_path = '/DATA/pranta_2411ai09/DialogueSummarization/data/samsum_pt_dataset'
print(f"Loading preprocessed dataset from: {dataset_path}")

# Load the tokenized dataset from disk
dataset_samsum_pt = load_from_disk(dataset_path)
print(f"Dataset loaded. Train size: {len(dataset_samsum_pt['train'])}, Validation size: {len(dataset_samsum_pt['validation'])}")

# Create a data collator: This dynamically pads the inputs to the longest 
# sequence in the batch for efficiency during sequence-to-sequence training.
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
print("Data Collator initialized for Seq2Seq padding.")

# ----------------------------------------------------------------------
# 3. CONFIGURE TRAINING ARGUMENTS
# ----------------------------------------------------------------------

# Define hyperparameters and configuration for the Trainer
print(f"--- 2. Configuring Training Parameters ---")
trainer_args = TrainingArguments(
    output_dir='./pegasus-samsum',                    # Directory where model checkpoints and logs will be saved
    num_train_epochs=5,                               # Total number of training epochs
    warmup_steps=500,
    fp16=True,                                 # Number of steps for learning rate warmup
    per_device_train_batch_size=4,                   # Batch size per GPU/device for training
    per_device_eval_batch_size=4,                    # Batch size per GPU/device for evaluation
    weight_decay=0.01,                                # Strength of weight decay (L2 regularization)
    logging_steps=10,                                 # Log training metrics every N steps
    evaluation_strategy='steps',                      # Evaluation is done at specified step intervals
    eval_steps=500,                                   # Run evaluation every 500 steps
    save_steps=1e6,                                   # Do not save checkpoints often (high value used here)
    gradient_accumulation_steps=16                    # Accumulate gradients over 16 steps to simulate a large batch size
) 
print(f"Training Arguments configured. Output directory: {trainer_args.output_dir}")

# ----------------------------------------------------------------------
# 4. INITIALIZE THE TRAINER
# ----------------------------------------------------------------------

print(f"--- 3. Initializing Trainer ---")
trainer = Trainer(
    model=model_pegasus, 
    args=trainer_args,
    tokenizer=tokenizer, 
    data_collator=seq2seq_data_collator,
    train_dataset=dataset_samsum_pt["train"], 
    eval_dataset=dataset_samsum_pt["validation"]
)
print("Trainer successfully initialized with model, data, and arguments.")

# ----------------------------------------------------------------------
# 5. START TRAINING
# ----------------------------------------------------------------------

print("===================================")
print(">>> TRAINING STARTED <<<")
print("===================================")

# Execute the fine-tuning process
trainer.train()

print("===================================")
print(">>> TRAINING COMPLETED <<<")
print("===================================")

# Save the final trained model and tokenizer
trainer.save_model('./pegasus-samsum')  # or any directory you want
# Save the tokenizer as well
tokenizer.save_pretrained('./pegasus-samsum')
print("===================================")
print(">>> MODEL SAVED <<<")
print("===================================")