# fine_tuning.py
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

def load_data(file_path):
    # Load and preprocess your dataset
    df = pd.read_csv(file_path)
    return df['Impression'].tolist()  # Example: use the 'Impression' column

def fine_tune_model(data):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Tokenize the data
    encodings = tokenizer('\n\n'.join(data), return_tensors='pt', truncation=True, padding=True)
    
    training_args = TrainingArguments(output_dir='./results', per_device_train_batch_size=2, num_train_epochs=3)
    trainer = Trainer(model=model, args=training_args, train_dataset=encodings)
    
    trainer.train()
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

if __name__ == "__main__":
    data = load_data('path_to_your_data.csv')
    fine_tune_model(data)
