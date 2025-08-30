import os
import requests
from control_panel.missions import SUPPORTED_MISSIONS

missions_list = "\n".join([f"- {m}" for m in SUPPORTED_MISSIONS.keys()])

SYSTEM_PROMPT = f"""
You are *RoboBartender*. Your job is to:
Greet customers, chat naturally, offer drinks and enter them to the queue.
Your supported drinks are {missions_list}
When users request drinks from the supported list, respond warmly and then issue commands for requested drinks of current order in this format: <drink '{{drink_name}}' count {{number}}>
Copy the exact text from supported drinks for commands (case sensitive)
When users say casual names, map them to exact mission names:
    for example: User says "beer" → Use 'pour beer' in command
   If user specifies amounts use those numbers, else order 1
For unsupported drinks Politely decline and suggest available options
**Multiple drinks:**
User: Can I get two margaritas and three beers?
RoboBartender: Absolutely! Getting those ready for you now. <drink 'example drink' count 2> <drink 'example drink' count 3>
"""

class LLMClient:
    def __init__(self, mode='openai'):
        """Initialize LLMClient with API mode and URLs."""
        self.mode = mode
        if mode == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
            self.api_url = 'https://api.openai.com/v1/chat/completions'
        else:
            self.api_key = 'dummy-key'
            self.api_url = 'http://localhost:4141/v1/messages?beta=true'
        self.headers = {'Authorization': f'Bearer {self.api_key}'}
        self.data = {
            'model': 'gpt-4o',
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT}
            ]
        }

    def send_message(self, user_message):
        """Send message to LLM and return response."""

        self.data['messages'].append({'role': 'user', 'content': user_message})
        response = requests.post(self.api_url, json=self.data, headers=self.headers)
        return response.json()
     
