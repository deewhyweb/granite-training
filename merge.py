

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "ibm-granite/granite-3.0-8b-instruct"  # Replace with your base model name
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

lora_model_path_1 = "./results"


peft_config_1 = PeftConfig.from_pretrained(lora_model_path_1)


lora_model_1 = PeftModel.from_pretrained(model, lora_model_path_1, peft_config=peft_config_1)

model = PeftModel.from_pretrained(base_model, lora_model_1)
merged_model = model.merge_and_unload()


merged_model.save_pretrained("./merged")