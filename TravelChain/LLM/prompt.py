from typing import List, Dict

from LLM.format import InputFormat, OutputFormat, Context

class Prompt():
    introductions: List[str]
    examples: List[str]
    
    #introduction, examples, context, input
    def create_messages(self, input: InputFormat, context: Context) -> List[Dict[str, str]]:
        messages:List[Dict[str,str]] = []
        
        messages.extend({"role": "system", "content": self.introductions})
        messages.extend({"role": "system", "content": self.examples})
        
        for chat in context.chats:
            chatting = [{"role": "system", "content" : chat.model_output}, 
                       {"role": "user", "content" : chat.user_input}]
            messages.extend(chatting)
        
        return messages
    
    def get_input_from_json(self, json_data) -> InputFormat:
        return InputFormat.model_validate_json(json_data)
    
    def get_json_from_input(self, input: InputFormat):
        return InputFormat.model_dump_json(input)
    
    def get_json_from_output(self, output: OutputFormat):
        return OutputFormat.model_dump_json(output)