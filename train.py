import transformers

transformers.set_seed(42)

import timeit

start_time = timeit.default_timer()
from datasets import load_dataset # type: ignore

#dataset = load_dataset('alespalla/chatbot_instruction_prompts')
dataset = load_dataset('json', data_files='train_Mixtral-8x7B-Instruct-v0_2024-11-12T15_54_01.jsonl')
test_dataset = dataset = load_dataset('json', data_files='test_Mixtral-8x7B-Instruct-v0_2024-11-12T15_54_01.jsonl')
# split_dataset = dataset['train'].train_test_split(test_size=0.2)
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

# def pirateify(batch):
#   prompts = [f"make it sound like a pirate said this, do not include any preamble or explanation only piratify the following: {response}" for response in batch['output']]
#   # Tokenize the inputs in batch and move them to GPU
#   inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')
#   # Generate the pirate-like responses in batch
#   outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.7)
#   # Decode the generated tokens into text for each output in the batch
#   pirate_responses = []
#   for output in outputs:
#     pr = tokenizer.decode(output, skip_special_tokens=True)
#     if '\n\n' in pr:
#       pirate_responses.append(pr.split('\n\n')[-1])
#     else:
#       pirate_responses.append(pr)

#   # Move the outputs back to CPU (to free up GPU memory)
#   inputs = inputs.to('cpu')
#   outputs = outputs.to('cpu')
#   # Clear the GPU cache to release any unused memory
#   torch.cuda.empty_cache()
#   return {
#       'prompt': batch['instruction'],  # The original prompts (already a batch)
#       'response': pirate_responses  # The pirate responses, generated in batch
#   }


def filter_long_examples(example):
    prompt_tokens = tokenizer.tokenize(example['user'])
    response_tokens = tokenizer.tokenize(example['assistant'])  # Tokenize the response
    return len(response_tokens) <= 200 and len(prompt_tokens) <= 50

# Apply the filter to both train and test splits
train_filtered = dataset['train']
test_filtered = test_dataset['train']

print(f"train_filtered: {len(train_filtered)} observations\ntest_filtered: {len(test_filtered)} observations")
# train_data = train_filtered.select(range(20)).map(pirateify, batched=True, batch_size=128)
# test_data = test_filtered.select(range(5)).map(pirateify, batched=True, batch_size=128)

# Save the new dataset
ft_dataset = datasets.DatasetDict({
    'train': train_filtered,
    'test': test_filtered
})
ft_dataset['train'].to_pandas().head()
import torch # type: ignore
torch.cuda.empty_cache()
start_time = timeit.default_timer()
input_text = "<|user>Who founded Parasol Insurance?\n<|assistant|>\n"

inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

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
    hub_model_id="rawkintrevo/granite-3.0-2b-instruct-pirate-adapter",
    learning_rate=2e-4,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=3,
    logging_steps=100,
    fp16=True,
    report_to="none"
)

max_seq_length = 250

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