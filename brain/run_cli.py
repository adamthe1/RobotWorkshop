import typer
from control_panel.LLM import LLMClient
import re
from logger_config import get_logger
from control_panel.mission_manager import MissionManager
from control_panel.robot_queue_locks import QueueClient
import os
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer()

@app.command()
def order_drink():
    """Chat with the robot bartender and order a drink."""

    client = LLMClient()
    MAX_JOBS = 3
    typer.echo('Welcome to RoboBartender!')
    with QueueClient() as queue:
        while True:
            queue_status = queue.get_status()
            logger = get_logger('CLI')
            logger.info(f"Queue status: {queue_status}")
            user_input = typer.prompt('What would you like to say?')

            response = client.send_message(user_input)
            # Prefer OpenAI style, fallback to your backend
            try:
                message = response['choices'][0]['message']['content']
            except (KeyError, IndexError):
                message = response.get('content',[{'text':''}])[0]['text']
            
            # Remove <create drink ...> before displaying to user
            logger.info(f"answered: {message}")
            visible_message = re.sub(r'<create drink ["\'](.+?)["\']>', '', message).strip()
            typer.echo(f'Bartender: {visible_message}')

            match = re.search(r'<create drink ["\'](.+?)["\'] count (\d+)>', message)
            if match:
                
                drink = match.group(1)
                count = match.group(2)

                for _ in range(int(count)):
                    if queue_status['mission_queue_length'] >= int(os.getenv('MISSION_QUEUE_SIZE', MAX_JOBS)):
                        typer.echo("Sorry, I can't take more drink orders right now. Try again later")
                    else:
                        typer.echo(f'Order received for: {drink}. Sending to robot...')
                        queue.enqueue_mission(drink)
                        queue_status = queue.get_status()

if __name__ == "__main__":
    app()