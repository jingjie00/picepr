from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load the base model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

# Tokenizer padding and truncation
def preprocess_data(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )

# Load and preprocess dataset
dataset = load_dataset("____", split="train")
dataset = dataset.map(preprocess_data, batched=True)

# LoRA configuration for PEFT
lora_config = LoraConfig(
    r=8,  # Rank of the adaptation matrices
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Fine-tune specific layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral-fine-tuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    fp16=True,  # Mixed precision training
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Start fine-tuning
trainer.train()

print("Start")
# Save the fine-tuned model
model.save_pretrained("./mistral-fine-tuned")
tokenizer.save_pretrained("./mistral-fine-tuned")
