import typer
from control_panel.LLM import LLMClient
import re

app = typer.Typer()

@app.command()
def order_drink():
    """Chat with the robot bartender and order a drink."""
    client = LLMClient()
    from control_panel.mission_manager import MissionManager
    MAX_JOBS = 1  # set your max jobs
    mission_manager = MissionManager()
    typer.echo('Welcome to RoboBartender!')
    with open('conversation.log', 'a') as log:
        while True:
            user_input = typer.prompt('What would you like to say?')
            response = client.send_message(user_input)
            # Prefer OpenAI style, fallback to your backend
            try:
                message = response['choices'][0]['message']['content']
            except (KeyError, IndexError):
                message = response.get('content',[{'text':''}])[0]['text']
            # Remove <create drink ...> before displaying to user
            visible_message = re.sub(r'<create drink ["\'](.+?)["\']>', '', message).strip()
            typer.echo(f'Bartender: {visible_message}')
            
            match = re.search(r'<create drink ["\'](.+?)["\']>', message)
            if match:
                drink = match.group(1)
                if mission_manager.mission_queue.qsize() >= MAX_JOBS:
                    typer.echo("Sorry, I can't take more drink orders right now. I'll update you when I can!")
                else:
                    log.write(f'User: {user_input}\nBartender: {message}\n')
                    typer.echo(f'Order received for: {drink}. Sending to robot...')
                    mission_manager.add_mission_to_queue(drink)

if __name__ == "__main__":
    app()