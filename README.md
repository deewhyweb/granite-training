pip install transformers datasets accelerate bitsandbytes peft trl

vllm serve ibm-granite/granite-3.0-8b-instruct \
    --enable-lora \
    --lora-modules parasol-lora=/home/instruct/granite-training/results


    curl http://0.0.0.0:8000/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{
      "model": "parasol-lora",
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant. "
        },
        {
          "role": "user", "content": "What is the history of Parasol Insurance?"
        }
      ]
    }'