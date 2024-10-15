from openai import OpenAI

from llm.format import InputFormat, OutputFormat, Context
from llm.prompt import Prompt

class Infer_Tool():
    client: OpenAI
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
        )
        
    def inference(self, input: InputFormat, context: Context):
        messages = Prompt.create_messages(input, context)
        
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=OutputFormat
        )
        
        output = completion.choices[0].message.parsed
        
        return output
        
    