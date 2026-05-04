from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

class MessageType(Enum):
    QUERY = "query"          # Query request
    RESPONSE = "response"    # Normal response
    ERROR = "error"         # Error information
    RETRY = "retry"         # Retry request
    FEEDBACK = "feedback"    # Feedback information

@dataclass
class Message:
    sender: str
    receiver: str
    content: Any
    msg_type: MessageType
    metadata: Optional[Dict] = None
    timestamp: float = datetime.now().timestamp()

    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.msg_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

class BaseAgent(ABC):
    def __init__(self, agent_id: str, llm_model: str = "gemini-2.5-flash"):
        self.agent_id = agent_id
        self.llm_model = llm_model
        self.message_queue: List[Message] = []
        self.logger = logging.getLogger(agent_id)

    def send_message(self, receiver: str, content: Any, msg_type: MessageType, metadata: Optional[Dict] = None) -> None:
        message = Message(self.agent_id, receiver, content, msg_type, metadata)
        self.message_queue.append(message)
        self.logger.info(f"Sent {msg_type.value} message to {receiver}")
        self.logger.debug(f"Message content: {content}")

    def receive_message(self, message: Message) -> Optional[Message]:
        if message.receiver == self.agent_id:
            self.logger.info(f"Received {message.msg_type.value} message from {message.sender}")
            self.logger.debug(f"Message content: {message.content}")
            return self.process_message(message)
        return None

    def get_messages(self, msg_type: Optional[MessageType] = None) -> List[Message]:
        if msg_type is None:
            return self.message_queue
        return [msg for msg in self.message_queue if msg.msg_type == msg_type]

    def clear_messages(self) -> None:
        self.message_queue.clear()

    @abstractmethod
    def process_message(self, message: Message) -> Optional[Message]:
        pass

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        error_msg = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        self.logger.error(f"Error in {self.agent_id}: {error_msg}")
        self.send_message(
            "coordinator",
            error_msg,
            MessageType.ERROR,
            {"timestamp": datetime.now().timestamp()}
        )