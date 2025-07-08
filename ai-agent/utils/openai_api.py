"""
OpenAI API wrapper for GPT-4 chat completions.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API with GPT-4 chat completions."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model: Model to use for completions
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        logger.info(f"Initialized OpenAI client with model: {model}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using GPT-4.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            system_message: Optional system message to prepend
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Prepare messages
            full_messages = []
            
            if system_message:
                full_messages.append({"role": "system", "content": system_message})
            
            full_messages.extend(messages)
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response data
            result = {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model
            }
            
            logger.info(f"Chat completion successful. Tokens used: {result['usage']['total_tokens']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    def create_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Create embeddings for text using OpenAI's embedding model.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            List of embedding values
        """
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            
            embedding = response.data[0].embedding
            logger.info(f"Created embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def reasoning_prompt(
        self,
        user_input: str,
        relevant_memories: List[str],
        agent_context: Dict[str, Any],
        task_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate a reasoning response based on user input and context.
        
        Args:
            user_input: The user's input/query
            relevant_memories: List of relevant memories from Pinecone
            agent_context: Additional context about the agent
            task_type: Type of task being performed
            
        Returns:
            Dictionary containing the reasoning response
        """
        system_message = self._build_system_message(agent_context)
        
        # Build context from memories
        memory_context = "\n".join([f"- {memory}" for memory in relevant_memories])
        
        user_message = f"""
Task Type: {task_type}

User Input: {user_input}

Relevant Memories:
{memory_context if memory_context else "No relevant memories found."}

Please analyze the user input, consider the relevant memories, and provide:
1. Your understanding of what the user is asking
2. Your reasoning process
3. Your response or action plan
4. Any follow-up questions or clarifications needed

Format your response as a structured analysis.
"""

        messages = [{"role": "user", "content": user_message}]
        
        return self.chat_completion(
            messages=messages,
            system_message=system_message,
            temperature=0.7,
            max_tokens=1000
        )
    
    def _build_system_message(self, agent_context: Dict[str, Any]) -> str:
        """Build system message from agent context."""
        return f"""You are {agent_context.get('name', 'an AI agent')} with the following capabilities:

Goals:
{chr(10).join([f"- {goal}" for goal in agent_context.get('goals', [])])}

You have access to long-term memory that helps you maintain context across interactions.
You should be helpful, accurate, and thoughtful in your responses.
When you're uncertain, acknowledge it and ask for clarification.

Current session context: {agent_context.get('description', 'General assistance')}
"""