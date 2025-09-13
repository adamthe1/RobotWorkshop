import os
import requests
from control_panel.missions import SUPPORTED_MISSIONS

SYSTEM_PROMPT = f"""
You are *RoboBartender*. Your job is to:
Greet customers, chat naturally, offer drinks and enter them to the queue.
When users request drinks from the supported list, respond warmly and then issue commands for requested drinks of current order in this format: <drink '{{drink_name}}' count {{number}}>
Copy the exact text from supported drinks for commands (case sensitive)
When users say casual names, map them to exact mission names:
    for example: User says "beer" → Use 'pour beer' in command
    User says "place cup on bar" → Use 'placeCupOnBar' in command
    User says "put cup on bar" → Use 'placeCupOnBar' in command
   If user specifies amounts use those numbers, else order 1
For unsupported drinks Politely decline and suggest available options
**Multiple drinks:**
User: Can I get two margaritas and three beers?
RoboBartender: Absolutely! Getting those ready for you now. <drink 'example drink' count 2> <drink 'example drink' count 3>

"""

class LLMClient:
    def __init__(self, possible_missions, mode='other'):
        """Initialize LLMClient with API mode and URLs."""
        self.mode = mode
        if mode == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
            self.api_url = 'https://api.openai.com/v1/chat/completions'
            self.model = 'gpt-4o'
        elif mode == "local":
            self.api_key = 'dummy-key'
            self.api_url = 'http://localhost:4141/v1/messages?beta=true'
            self.model = 'gpt-4o'
        else: 
            self.api_key = os.getenv('GLM_API_KEY')
            self.api_url = 'https://api.z.ai/api/paas/v4/chat/completions'
            self.model = 'glm-4.5'
        self.headers = {'Authorization': f'Bearer {self.api_key}'}
        self.SYSTEM_PROMPT = SYSTEM_PROMPT + f"\nSupported drinks are: {', '.join(possible_missions)}"
        self.data = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self.SYSTEM_PROMPT}
            ]
        }

    def send_message(self, user_message):
        """Send message to LLM and return response."""

        self.data['messages'].append({'role': 'user', 'content': user_message})
        response = requests.post(self.api_url, json=self.data, headers=self.headers)
        return response.json()

    # -------- Health / configuration checks --------
    def is_configured_and_healthy(self, timeout: float = 3.0) -> bool:
        """Quickly validate LLM configuration by sending a tiny ping.

        Returns False if API key/url is missing, the request errors or
        the response structure is clearly invalid.
        """
        # Basic presence checks
        if not self.api_url:
            return False
        if not self.api_key:
            return False

        # Build a minimal payload without mutating the main conversation
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': 'You are a health check.'},
                {'role': 'user', 'content': 'ping'}
            ]
        }
        try:
            resp = requests.post(self.api_url, json=payload, headers=self.headers, timeout=timeout)
            if resp.status_code >= 400:
                return False
            data = resp.json()
            # Try to detect a plausible content field
            content = None
            try:
                content = data['choices'][0]['message']['content']
            except Exception:
                try:
                    content = data.get('content', [{'text': ''}])[0]['text']
                except Exception:
                    content = None
            return isinstance(content, str)
        except Exception:
            return False

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test LLMClient")
    parser.add_argument('--mode', choices=['openai', 'local', 'other'], default='openai', help='API mode')
    parser.add_argument('--message', default='Hello, please respond briefly.', help='Message to send to the LLM')
    args = parser.parse_args()

    client = LLMClient(mode='other')
    try:
        resp = client.send_message(args.message)
        print(json.dumps(resp, indent=2))
    except Exception as e:
        print("Error calling LLM:", e)

     
