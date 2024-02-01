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
# Note: if you get errors about mistral-instruct not being a valid chat format,
# update your llama-cpp-python package
llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=32768, n_threads=8, echo=False, chat_format="mistral-instruct")

# Record the end time for model loading
end_time = time.time()

# Log the time taken to load the model
print(f'Model load took {end_time - start_time} seconds')    

successful_runs = 0
total_runs = 100

for i in range(total_runs):
    try:
        # Initialize the chat.
        output = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "Act as an AI that is designed to output valid JSON. Generate a demo JSON object."
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )

        # Verify that the output is a valid JSON object
        json.loads(output["choices"][-1]["message"]["content"])
        
        successful_runs += 1
    except Exception as e:
        print(output["choices"][-1]["message"]["content"])
        
print(f'Successful runs: {successful_runs}/{total_runs}')