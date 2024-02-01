import json
import time
from llama_cpp import Llama

# Path to the model file
model_path = 'C:\\Users\\kenny\\.cache\\lm-studio\\models\\TheBloke\\Mistral-7B-Instruct-v0.2-GGUF\\mistral-7b-instruct-v0.2.Q4_K_M.gguf'

# Initialize the logging
print('Loading model...')

# Record the start time for model loading
start_time = time.time()

# Create a Llama model instance.
llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=32768, n_threads=8, echo=False)

# Record the end time for model loading
end_time = time.time()

# Log the time taken to load the model
print(f'Model load took {end_time - start_time} seconds')

# Format the instruction in Mistral-Instruct format
json_start = "{\n\t"
json_end = "\n}"
instruction = "<s>[INST]Generate a demo JSON object.[/INST]\n" + json_start

# Create a completion
output = llm.create_completion(
    prompt=instruction,
    max_tokens=1000,
    temperature=0.3,
    stop=json_end
)

# Extract the generated content from the output
generated_content = json_start + output["choices"][0]["text"] + json_end

# Verify that the output is a valid JSON object
print(json.dumps(json.loads(generated_content), indent=4))