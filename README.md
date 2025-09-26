# Dialogue Summarization with Pegasus

## Overview
This project implements **abstractive dialogue summarization** using the Pegasus model from Hugging Face Transformers. The goal is to generate concise, human-like summaries for multi-turn dialogues, leveraging the SAMSum dataset for training and evaluation.

## Features
- **Preprocessing:** Tokenizes and prepares dialogue data for model training.
- **Model Training:** Fine-tunes Pegasus on the SAMSum dataset using PyTorch and Hugging Face Trainer.
- **Evaluation:** Computes ROUGE metrics to assess summary quality and supports custom decoding strategies (beam search, length penalty).
- **Inference:** Generates summaries for new dialogues and compares them to reference summaries.

## Dataset
- **SAMSum Corpus:** A human-annotated dataset for dialogue summarization, containing thousands of real-world chat dialogues and their summaries.
- **Format:** Each sample includes an `id`, `dialogue` (multi-turn text), and `summary` (human-written).

## Project Structure
```
DialogueSummarization/
├── data/                # Preprocessed and raw datasets
├── src/                 # Source code for training, evaluation, and inference
│   ├── 01_preprocessing.py
│   ├── 02_training.py
│   └── 03_evaluation.py
├── pegasus-samsum/      # Saved model checkpoints and tokenizer
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

## Quick Start
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Preprocess the dataset:**
   ```bash
   python src/01_preprocessing.py
   ```
3. **Train the model:**
   ```bash
   python src/02_training.py
   ```
4. **Evaluate the model:**
   ```bash
   python src/03_evaluation.py
   ```

## Model Usage Example
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('./pegasus-samsum')
tokenizer = AutoTokenizer.from_pretrained('./pegasus-samsum')
dialogue = "Hey, are you coming to the meeting?\nYes, I'll be there in 10 minutes.\nGreat, see you soon!"
inputs = tokenizer(dialogue, return_tensors='pt', truncation=True, max_length=1024)
summary_ids = model.generate(inputs['input_ids'], num_beams=4, length_penalty=1.0, max_length=60, early_stopping=True)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

## Evaluation Metrics
- **ROUGE-1, ROUGE-2, ROUGE-L:** Used to measure overlap between generated and reference summaries.
- Typical scores for Pegasus on SAMSum: ROUGE-1 ~0.45–0.50, ROUGE-2 ~0.20–0.25, ROUGE-L ~0.35–0.40.

## Customization
- **Decoding strategies:** Adjust `num_beams`, `length_penalty`, and other parameters in `model.generate()` for best results.
- **Fine-tuning:** You can further fine-tune the model on your own dialogue data for improved performance.

## License
This project uses the SAMSum dataset for research purposes. Please refer to the dataset's license for usage restrictions.

## References
- [SAMSum Corpus Paper](https://arxiv.org/abs/1911.12237)
- [Hugging Face Pegasus Documentation](https://huggingface.co/docs/transformers/en/model_doc/pegasus)
- https://www.youtube.com/watch?v=p7V4Aa7qEpw&t=4842s&pp=ygUWbmxwIGVuZCB0byBlbmQgcHJvamVjdA%3D%3D
- https://github.com/entbappy/Branching-tutorial/raw/master/summarizer-data.zip
