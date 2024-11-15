"""
Set up this example by starting a vLLM OpenAI-compatible server with tool call
options enabled. For example:

IMPORTANT: for mistral, you must use one of the provided mistral tool call
templates, or your own - the model default doesn't work for tool calls with vLLM
See the vLLM docs on OpenAI server & tool calling for more details.

sudo dnf -y install libcudnn8 libcudnn8-devel cuda-cccl-12-4 libnccl-2.22.3-1+cuda12.4.x86_64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64:/home/instruct/.local/lib/python3.11/site-packages/nvidia/cudnn/lib
pip install git+https://github.com/vllm-project/vllm.git

vllm serve ibm-granite/granite-3.0-8b-instruct \
            --chat-template granite.jinja \
             --tool-call-parser granite --enable-auto-tool-choice


"""
import json

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type":
                    "string",
                    "description":
                    "The city to find the weather for, e.g. 'San Francisco'"
                },
                "state": {
                    "type":
                    "string",
                    "description":
                    "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city", "state", "unit"]
        }
    }
}]

messages = [{
    "role": "user",
    "content": "Hi! How are you doing today?"
}, {
    "role": "assistant",
    "content": "I'm doing well! How can I help you?"
}, {
    "role":
    "user",
    "content":
    "Can you tell me what the temperate will be in Dallas, in fahrenheit?"
}]

chat_completion = client.chat.completions.create(messages=messages,
                                                 model=model,
                                                 tools=tools)

print("Chat completion results:")
print(chat_completion)
print("\n\n")

tool_calls_stream = client.chat.completions.create(messages=messages,
                                                   model=model,
                                                   tools=tools,
                                                   stream=True)

chunks = []
for chunk in tool_calls_stream:
    chunks.append(chunk)
    if chunk.choices[0].delta.tool_calls:
        print(chunk.choices[0].delta.tool_calls[0])
    else:
        print(chunk.choices[0].delta)

arguments = []
tool_call_idx = -1
for chunk in chunks:

    if chunk.choices[0].delta.tool_calls:
        tool_call = chunk.choices[0].delta.tool_calls[0]

        if tool_call.index != tool_call_idx:
            if tool_call_idx >= 0:
                print(
                    f"streamed tool call arguments: {arguments[tool_call_idx]}"
                )
            tool_call_idx = chunk.choices[0].delta.tool_calls[0].index
            arguments.append("")
        if tool_call.id:
            print(f"streamed tool call id: {tool_call.id} ")

        if tool_call.function:
            if tool_call.function.name:
                print(f"streamed tool call name: {tool_call.function.name}")

            if tool_call.function.arguments:
                arguments[tool_call_idx] += tool_call.function.arguments

if len(arguments):
    print(f"streamed tool call arguments: {arguments[-1]}")

print("\n\n")

messages.append({
    "role": "assistant",
    "tool_calls": chat_completion.choices[0].message.tool_calls
})


# Now, simulate a tool call
def get_current_weather(city: str, state: str, unit: 'str'):
    return ("The weather in Dallas, Texas is 85 degrees fahrenheit. It is "
            "partly cloudly, with highs in the 90's.")


available_tools = {"get_current_weather": get_current_weather}

completion_tool_calls = chat_completion.choices[0].message.tool_calls
for call in completion_tool_calls:
    tool_to_call = available_tools[call.function.name]
    args = json.loads(call.function.arguments)
    result = tool_to_call(**args)
    print(result)
    messages.append({
        "role": "tool",
        "content": result,
        "tool_call_id": call.id,
        "name": call.function.name
    })

chat_completion_2 = client.chat.completions.create(messages=messages,
                                                   model=model,
                                                   tools=tools,
                                                   stream=False)
print("\n\n")
print(chat_completion_2)