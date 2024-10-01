from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load data and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Fine-tuning steps, load your dataset here
training_args = TrainingArguments(output_dir='./results', per_device_train_batch_size=2, num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=eval_data)

trainer.train()
