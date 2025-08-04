from LLM import LLMClient

if __name__ == "__main__":
    client = LLMClient()
    while True:
        user_message = input('User: ')
        response = client.send_message(user_message)
        print('LLM response:', response)
