import os
import requests
from control_panel.system_prompt import SYSTEM_PROMPT

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
