import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define device as 'cuda' if a GPU is available for faster computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model and tokenizer paths
model_path = "./merged"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model; set device_map based on your setup
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()

# Define a user query in a structured format
chat = [
    { "role": "user", "content": "Who founded Parasol Insurance?" },
]

# Prepare the chat data with the required prompts
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Tokenize the input chat
input_tokens = tokenizer(chat, return_tensors="pt").to(device)
# Generate output tokens with a maximum of 100 new tokens in the response
output = model.generate(**input_tokens, max_new_tokens=100)
# Decode and print the response
response = tokenizer.batch_decode(output, skip_special_tokens=True)
print(response[0])