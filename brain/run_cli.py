import typer
from control_panel.LLM import LLMClient
import re
from logger_config import get_logger
from control_panel.mission_manager import MissionManager

app = typer.Typer()

@app.command()
def order_drink():
    """Chat with the robot bartender and order a drink."""

    client = LLMClient()
    MAX_JOBS = 1
    typer.echo('Welcome to RoboBartender!')
    
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
            logger.info(f"Detected drink order: {drink}")
            
            if robot_queue.queues['robot_1'].qsize() >= MAX_JOBS:
                logger.warning(f"Queue full, rejecting drink order: {drink}")
                typer.echo("Sorry, I can't take more drink orders right now. I'll update you when I can!")
            else:
                logger.info(f"Processing drink order: {drink}")
                typer.echo(f'Order received for: {drink}. Sending to robot...')
                robot_queue.enqueue_mission('robot_1', drink)
                logger.info(f"Successfully queued drink order: {drink}")

if __name__ == "__main__":
    app()