import typer
from control_panel.LLM import LLMClient
import re
from logger_config import get_logger
from control_panel.mission_manager import MissionManager
from control_panel.robot_queue_locks import QueueClient
import os
from dotenv import load_dotenv
from control_panel.missions import SUPPORTED_MISSIONS
from typing import List, Tuple, Optional

load_dotenv()

app = typer.Typer()

class DrinkOrderProcessor:
    """Handles drink order parsing, validation, and processing"""
    
    def __init__(self, queue_client: QueueClient, max_queue_size: int):
        self.queue_client = queue_client
        self.max_queue_size = max_queue_size
        self.logger = get_logger('DrinkOrderProcessor')

    def parse_drink_orders(self, message: str) -> List[Tuple[str, int]]:
        """Parse drink orders from LLM message"""
        drink_matches = re.findall(r'<drink ["\'](.+?)["\'] count (\d+)>', message)
        return [(drink, int(count)) for drink, count in drink_matches]

    def validate_drinks(self, orders: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Validate drinks against supported missions"""
        valid_orders = []
        unsupported_drinks = []
        
        for drink, count in orders:
            if drink in SUPPORTED_MISSIONS:
                valid_orders.append((drink, count))
            else:
                unsupported_drinks.append(drink)
        
        return valid_orders, unsupported_drinks

    def process_orders(self, valid_orders: List[Tuple[str, int]]) -> int:
        """Process valid drink orders and add to queue"""
        total_drinks_ordered = 0
        
        for drink, count in valid_orders:
            for _ in range(count):
                current_queue_status = self.queue_client.get_status()
                if current_queue_status['mission_queue_length'] >= self.max_queue_size:
                    self.logger.info("Sorry, I can't take more drink orders right now. Try again later")
                    return total_drinks_ordered
                else:
                    self.logger.info(f'Order received for: {drink}')
                    self.queue_client.enqueue_mission(drink)
                    total_drinks_ordered += 1
        
        return total_drinks_ordered

class LLMResponseHandler:
    """Handles LLM communication and response processing"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = get_logger('LLMResponseHandler')

    def get_llm_response(self, user_input: str) -> str:
        """Get response from LLM and extract message content"""
        response = self.llm_client.send_message(user_input)
        
        # Prefer OpenAI style, fallback to your backend
        try:
            message = response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            message = response.get('content', [{'text': ''}])[0]['text']
        
        self.logger.info(f"LLM answered: {message}")
        return message

    def create_correction_prompt(self, unsupported_drinks: List[str]) -> str:
        """Create correction prompt for unsupported drinks"""
        unsupported_list = ", ".join(unsupported_drinks)
        return (f"The drinks '{unsupported_list}' are not supported. "
                f"Please retry with supported drinks or ask for a better explanation of what you want. "
                f"Supported drinks include: {', '.join(SUPPORTED_MISSIONS)}, You must use the exact drink names as listed, ")

    def clean_message_for_display(self, message: str) -> str:
        """Remove drink tags from message for user display"""
        return re.sub(r'<drink ["\'](.+?)["\'] count (\d+)>', '', message).strip()

class DrinkOrderSession:
    """Main session handler for drink ordering"""
    
    def __init__(self):
        self.llm_handler = LLMResponseHandler(LLMClient())
        self.logger = get_logger('CLI')
        self.max_jobs = int(os.getenv('MISSION_QUEUE_SIZE', 3))

    def handle_user_input(self, user_input: str, queue_client: QueueClient) -> None:
        """Handle a single user input and process drink orders"""
        order_processor = DrinkOrderProcessor(queue_client, self.max_jobs)
        
        # Keep trying until we get valid drinks or no drinks
        current_input = user_input
        
        while True:
            message = self.llm_handler.get_llm_response(current_input)
            orders = order_processor.parse_drink_orders(message)
            
            if not orders:
                # No drink orders found, just show the message
                typer.echo(f'Bartender: {message}')
                break
            
            valid_orders, unsupported_drinks = order_processor.validate_drinks(orders)
            
            if unsupported_drinks:
                # Re-query LLM for unsupported drinks
                correction_prompt = self.llm_handler.create_correction_prompt(unsupported_drinks)
                self.logger.info(f"Correcting LLM with: {correction_prompt}")
                current_input = correction_prompt
                continue
            
            # All drinks are valid, process the orders
            visible_message = self.llm_handler.clean_message_for_display(message)
            typer.echo(f'Bartender: {visible_message}')
            
            total_drinks_ordered = order_processor.process_orders(valid_orders)
            
            if total_drinks_ordered > 0:
                typer.echo(f'Total drinks ordered: {total_drinks_ordered}. Sending to robots...')
            
            break

    def run_session(self, queue_client: QueueClient) -> None:
        """Run the main drink ordering session"""
        while True:
            queue_status = queue_client.get_status()
            self.logger.info(f"Queue status: {queue_status}")
            
            user_input = typer.prompt('What would you like to say?')
            self.handle_user_input(user_input, queue_client)

@app.command()
def order_drink():
    """Chat with the robot bartender and order a drink."""
    typer.echo('Welcome to RoboBartender!')
    
    session = DrinkOrderSession()
    
    with QueueClient() as queue:
        session.run_session(queue)

if __name__ == "__main__":
    app()