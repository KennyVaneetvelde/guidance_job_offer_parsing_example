import json

import guidance
from guidance import gen, select

@guidance
def gen_json_array_of_strings(lm, name, max=20):
    # Start the JSON array
    lm += '['
    i = 0
    while True:
        # Generate a string with the specified regex. The regex is used to ensure that the value only contains letters, spaces, punctuation, numbers, and dashes.
        lm += '"' + gen(name, list_append=True, regex=r'[a-zA-Z0-9\s\-\.\,\!\?\+\$]+', stop='"')
        
        # Select a comma or a closing bracket. Note that we re-include the quotation mark, as it is not included in the output of the gen function.
        lm += select(options=['",', '"]'], name='choice')
        
        # If the choice is a closing bracket, the array is complete and we can return the language model.
        if lm['choice'] == '"]':
            return lm
        
        i += 1
        if i == max:
            return lm

# Read job_offer.txt from the assets folder
job_offer = open('assets/job_offer.txt', 'r').read()

# Path to the model file
model_path = 'C:\\Users\\kenny\\.cache\\lm-studio\\models\\TheBloke\\Mistral-7B-Instruct-v0.2-GGUF\\mistral-7b-instruct-v0.2.Q4_K_M.gguf'
llm = guidance.models.LlamaCpp(model_path, n_gpu_layers=-1, n_ctx=32768, n_threads=8, echo=False)

# Let's say our database only contains these options for degrees.
degrees = [
    'Bachelor\'s Degree Preferred', 
    'Bachelor\'s Degree Required', 
    'Master\'s Degree Preferred', 
    'Master\'s Degree Required', 
    'PhD Preferred', 
    'PhD Required', 
    'No Degree Required'
]

# This regex is used to ensure that the value only contains letters and spaces. Totally optional but it's a good idea to constrain the output to the desired format.
string_regex = r'[a-zA-Z\s]+'

output = llm + f"""<s>[INST]====BEGIN EXTRA CONTEXT====
{job_offer}
====END EXTRA CONTEXT====
====BEGIN INSTRUCTION====
Convert the job offer to a JSON object.
====END INSTRUCTION====[/INST]
{{
  "jobTitle": "{gen('jobTitle', regex=string_regex, stop='"', max_tokens=10)}",
  "company": "{gen('company', regex=string_regex, stop='"', max_tokens=10)}",
  "location": "{gen('location', regex=string_regex, stop='"', max_tokens=10)}",
  "salary": {gen('salary', regex=r'[0-9]+', stop=',')},
  "shortDescription": "{gen('shortDescription', regex=string_regex, stop='"', max_tokens=100)}",
  "minimalEducationRequired": "{select(degrees, name='degree')}",
  "skillsRequired": {gen_json_array_of_strings('skills', max=10)},
  "extraBenefits": {gen_json_array_of_strings('benefits', max=10)}
}}
"""

# Extract the generated content from the output
output_json_string = str(output).split('[/INST]')[-1].strip()

# Create a completion
print(json.dumps(json.loads(output_json_string), indent=4))