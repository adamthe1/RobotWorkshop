import os
import requests
from .missions import SUPPORTED_MISSIONS

missions_list = "\n".join([f"- {m}" for m in SUPPORTED_MISSIONS.keys()])

SYSTEM_PROMPT = f"""
You are RoboBartender, an interactive robot bartender. Your job is to:

1. Greet the user and ask what they would like to drink.
2. If the user requests a drink available in the supported missions below, respond warmly ("Coming right up!", "On it!", etc.) and issue a command in this format: <create drink '{{drink}}'>.
3. Only accept requests for these drinks:
{missions_list}
4. If the drink is not supported, kindly suggest the available options.
5. Continue the conversation naturally until a drink request is made.
6. Once a valid drink request is confirmed, respond and issue the proper command only ONCE for each order.

Example drinks list:
{missions_list}
"""

class LLMClient:
    def __init__(self, mode='custom'):
        """Initialize LLMClient with API mode and URLs."""
        self.mode = mode
        if mode == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
            self.api_url = 'https://api.openai.com/v1/chat/completions'
        else:
            self.api_key = 'dummy-key'
            self.api_url = 'http://localhost:4141/v1/messages?beta=true'

    def send_message(self, user_message):
        """Send message to LLM and return response."""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        data = {
            'model': 'gpt-4o',
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_message}
            ]
        }
        response = requests.post(self.api_url, json=data, headers=headers)
        return response.json()

