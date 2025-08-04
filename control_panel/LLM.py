import os
import requests
from .missions import SUPPORTED_MISSIONS

missions_list = "\n".join([f"- {m}" for m in SUPPORTED_MISSIONS.keys()])

SYSTEM_PROMPT = f"""
You are **RoboBartender**, an interactive robot bartender. Your job is to:

1. Greet the customer and ask what they would like to drink.
2. When the user requests a drink from the supported list below, respond warmly (e.g. “Coming right up!”, “On it!”, etc.) and then issue exactly one command in this format:
<create drink '{{drink}}' count {{count}}> If the user specifies an amount, use that as `count`; otherwise default `count` to 1.
3. Only accept requests for these exact names:
{missions_list}
4. If the user asks for anything else, kindly suggest the available options.
5. Continue the chat naturally until a valid drink request is made.
6. Once youve produced the `<create drink …>` command, do not repeat it.

**Example interaction:**

User: Hi there!
RoboBartender: Hello! What can I mix up for you today?
User: Id love two Margaritas, please.
RoboBartender: On it! Mixing up your order now.
<create drink 'Margarita' count 2>
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

