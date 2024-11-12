import transformers

transformers.set_seed(42)

import timeit

start_time = timeit.default_timer()
from datasets import load_dataset # type: ignore

dataset = load_dataset('json', data_files='train_Mixtral-8x7B-Instruct-v0_2024-11-12T15_54_01.jsonl')
test_dataset = load_dataset('json', data_files='test_Mixtral-8x7B-Instruct-v0_2024-11-12T15_54_01.jsonl')

dataset_loadtime = timeit.default_timer() - start_time

start_time = timeit.default_timer()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig # type: ignore
from peft import LoraConfig
from trl import SFTTrainer

model_checkpoint = "ibm-granite/granite-3.0-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 # if not set will throw a warning about slow speeds when training
)

model = AutoModelForCausalLM.from_pretrained(
  model_checkpoint,
  quantization_config=bnb_config,
  device_map="auto"

)

model_loadtime = timeit.default_timer() - start_time

from transformers import pipeline # type: ignore
import datasets # type: ignore

# Apply the filter to both train and test splits
train_filtered = dataset['train']
test_filtered = test_dataset['train']

print(f"train_filtered: {len(train_filtered)} observations\ntest_filtered: {len(test_filtered)} observations")

# Save the new dataset
ft_dataset = datasets.DatasetDict({
    'train': train_filtered,
    'test': test_filtered
})
ft_dataset['train'].to_pandas().head()
import torch # type: ignore
torch.cuda.empty_cache()
start_time = timeit.default_timer()

model_check_loadtime = timeit.default_timer() - start_time

start_time = timeit.default_timer()
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['user'])):

        text = f"<|system|>\nYou are a helpful assistant\n<|user|>\n{example['user'][i]}\n<|assistant|>\n{example['assistant'][i]}<|endoftext|>"
        output_texts.append(text)
    return output_texts


response_template = "\n<|assistant|>\n"

from trl import DataCollatorForCompletionOnlyLM # type: ignore

response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


# Apply qLoRA
qlora_config = LoraConfig(
    r=16,  # The rank of the Low-Rank Adaptation
    lora_alpha=32,  # Scaling factor for the adapted layers
    target_modules=["q_proj", "v_proj"],  # Layer names to apply LoRA to
    lora_dropout=0.1,
    bias="none"
)

# Initialize the SFTTrainer
training_args = TrainingArguments(
    output_dir="./results",
    hub_model_id="rawkintrevo/granite-3.0-2b-instruct-adapter",
    learning_rate=2e-4,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=10,
    logging_steps=100,
    fp16=True,
    report_to="none"
)

max_seq_length = 2500

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ft_dataset['train'],
    eval_dataset=ft_dataset['test'],
    tokenizer=tokenizer,
    peft_config = qlora_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=max_seq_length,
)

training_setup_loadtime = timeit.default_timer() - start_time

start_time = timeit.default_timer()
# Start training
print("Starting training")
trainer.train()
training_time = timeit.default_timer() - start_time
trainer.save_model("./results")


input_text = "<|user>Who founded Parasol insurance?\n<|assistant|>\n"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
stop_token = "<|endoftext|>"
stop_token_id = tokenizer.encode(stop_token)[0]
outputs = model.generate(**inputs, max_new_tokens=500, eos_token_id=stop_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


input_ids= tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids=input_ids)
print(tokenizer.decode(outputs[0]))