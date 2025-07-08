"""
Main entry point for the Pinecone-powered AI Agent.
"""

import json
import os
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

# Import all agent components
from utils.logger import JSONLLogger
from utils.openai_api import OpenAIClient
from memory.memory_interface_pinecone import PineconeMemoryInterface
from reasoning.reasoning_engine import ReasoningEngine
from feedback.self_improvement import SelfImprovementEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAgent:
    """Main AI Agent class that orchestrates all components."""
    
    def __init__(self, config_path: str = "config/agent_config.json"):
        """
        Initialize the AI Agent with all its components.
        
        Args:
            config_path: Path to the agent configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize logger
        log_file = self.config.get("logging_settings", {}).get("log_file", "agent_interactions.jsonl")
        self.json_logger = JSONLLogger(log_file)
        
        # Initialize components
        self._initialize_components()
        
        logger.info("AI Agent initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            config_file = Path(__file__).parent / self.config_path
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Return default configuration
            return {
                "agent_name": "AI Agent",
                "goals": ["Assist users with their queries"],
                "memory_settings": {
                    "pinecone_index_name": "agent-memory-index",
                    "embedding_dimension": 1536,
                    "top_k_memories": 5,
                    "memory_relevance_threshold": 0.7
                },
                "reasoning_settings": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
    
    def _initialize_components(self):
        """Initialize all agent components."""
        try:
            # Initialize memory interface
            memory_settings = self.config.get("memory_settings", {})
            self.memory = PineconeMemoryInterface(
                index_name=memory_settings.get("pinecone_index_name", "agent-memory-index"),
                embedding_dimension=memory_settings.get("embedding_dimension", 1536)
            )
            
            # Initialize reasoning engine
            self.reasoning = ReasoningEngine(self.config, self.json_logger)
            
            # Initialize self-improvement engine
            self.self_improvement = SelfImprovementEngine(self.config, self.json_logger)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def run(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for processing user input.
        
        Args:
            input_dict: Dictionary containing user input and metadata
            
        Returns:
            Dictionary containing the agent's response and metadata
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not isinstance(input_dict, dict):
                raise ValueError("Input must be a dictionary")
            
            user_query = input_dict.get("query", "")
            if not user_query:
                raise ValueError("Query field is required in input")
            
            logger.info(f"Processing query: {user_query[:100]}...")
            
            # Step 1: Fetch relevant memories from Pinecone
            memories = self._fetch_memory(user_query, input_dict)
            
            # Step 2: Generate reasoning and decision using GPT-4
            reasoning_result = self._decide_next_action(input_dict, memories)
            
            # Step 3: Summarize interaction and store as memory
            memory_summary = self._summarize_interaction(
                input_dict, reasoning_result, memories, start_time
            )
            
            # Step 4: Store the interaction summary as a new memory
            self._store_interaction_memory(memory_summary)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Prepare final response
            response = {
                "success": reasoning_result.get("success", True),
                "response": reasoning_result.get("response", ""),
                "action": reasoning_result.get("action", "respond"),
                "confidence": reasoning_result.get("confidence", 0.8),
                "reasoning_steps": reasoning_result.get("reasoning_steps", []),
                "memories_used": len(memories),
                "execution_time": execution_time,
                "timestamp": memory_summary.get("timestamp"),
                "session_info": {
                    "agent_name": self.config.get("agent_name"),
                    "model_used": reasoning_result.get("model_used", "gpt-4"),
                    "tokens_used": reasoning_result.get("tokens_used", {}),
                    "memory_stats": self.memory.get_memory_stats()
                }
            }
            
            # Log the complete interaction
            self.json_logger.log_interaction(
                interaction_type=input_dict.get("type", "general"),
                user_input=input_dict,
                agent_response=response,
                memories_retrieved=[mem[0] for mem in memories],
                reasoning_process=reasoning_result,
                execution_time=execution_time
            )
            
            logger.info(f"Successfully processed query in {execution_time:.2f} seconds")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_response = {
                "success": False,
                "error": str(e),
                "response": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "action": "error_response",
                "confidence": 0.0,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
            
            # Log the error
            self.json_logger.log_interaction(
                interaction_type="error",
                user_input=input_dict,
                agent_response=error_response,
                execution_time=execution_time
            )
            
            logger.error(f"Error processing query: {e}")
            return error_response
    
    def _fetch_memory(self, query: str, input_dict: Dict[str, Any]) -> list:
        """Fetch relevant memories from Pinecone."""
        try:
            memory_settings = self.config.get("memory_settings", {})
            top_k = memory_settings.get("top_k_memories", 5)
            threshold = memory_settings.get("memory_relevance_threshold", 0.7)
            
            memories = self.memory.retrieve_memories(
                query=query,
                top_k=top_k,
                relevance_threshold=threshold
            )
            
            self.json_logger.log_memory_operation(
                operation="retrieve",
                details={
                    "query": query[:100],
                    "memories_found": len(memories),
                    "top_k": top_k,
                    "threshold": threshold
                },
                success=True
            )
            
            logger.info(f"Retrieved {len(memories)} relevant memories")
            return memories
            
        except Exception as e:
            self.json_logger.log_memory_operation(
                operation="retrieve",
                details={"query": query[:100]},
                success=False,
                error_message=str(e)
            )
            logger.error(f"Error fetching memories: {e}")
            return []
    
    def _decide_next_action(self, input_dict: Dict[str, Any], memories: list) -> Dict[str, Any]:
        """Use reasoning engine to decide next action."""
        try:
            result = self.reasoning.decide_next_action(
                user_input=input_dict,
                relevant_memories=memories,
                context={"agent_config": self.config}
            )
            return result
            
        except Exception as e:
            logger.error(f"Error in reasoning: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": "error_response",
                "response": "I encountered an error while reasoning about your request.",
                "confidence": 0.0
            }
    
    def _summarize_interaction(
        self,
        input_dict: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        memories: list,
        start_time: float
    ) -> Dict[str, Any]:
        """Summarize the interaction for storage."""
        try:
            execution_context = {
                "execution_time": time.time() - start_time,
                "tokens_used": reasoning_result.get("tokens_used", {}),
                "session_id": self.json_logger._get_session_id()
            }
            
            summary = self.self_improvement.summarize_interaction(
                user_input=input_dict,
                agent_response=reasoning_result,
                memories_used=[mem[0] for mem in memories],
                execution_context=execution_context
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing interaction: {e}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "summary": f"Error summarizing interaction: {str(e)}"
            }
    
    def _store_interaction_memory(self, summary: Dict[str, Any]):
        """Store interaction summary as memory in Pinecone."""
        try:
            if summary.get("success", True):
                memory_text = self.self_improvement.create_memory_summary(summary)
                
                memory_id = self.memory.store_memory(
                    content=memory_text,
                    memory_type="interaction",
                    metadata={
                        "interaction_type": summary.get("interaction_type", "general"),
                        "success": summary.get("success", True),
                        "confidence": summary.get("confidence", 0.8),
                        "session_id": summary.get("session_id", "unknown")
                    }
                )
                
                self.json_logger.log_memory_operation(
                    operation="store",
                    details={
                        "memory_id": memory_id,
                        "memory_type": "interaction",
                        "content_length": len(memory_text)
                    },
                    success=True
                )
                
                logger.info(f"Stored interaction memory with ID: {memory_id}")
            else:
                logger.info("Skipping memory storage for failed interaction")
                
        except Exception as e:
            self.json_logger.log_memory_operation(
                operation="store",
                details={"memory_type": "interaction"},
                success=False,
                error_message=str(e)
            )
            logger.error(f"Error storing interaction memory: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            return self.memory.get_memory_stats()
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}
    
    def analyze_performance(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze agent performance over a time window."""
        try:
            recent_logs = self.json_logger.read_recent_logs(limit=100)
            return self.self_improvement.analyze_performance_trends(
                recent_logs, time_window_hours
            )
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e), "status": "error"}


# Global agent instance
_agent_instance = None


def get_agent_instance(config_path: str = "config/agent_config.json") -> AIAgent:
    """Get or create the global agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AIAgent(config_path)
    return _agent_instance


def run_agent(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point function for running the AI agent.
    
    Args:
        input_dict: Dictionary containing user input and metadata.
                   Required fields:
                   - query: str - The user's query or input
                   Optional fields:
                   - type: str - Type of interaction (default: "general")
                   - context: dict - Additional context
                   - metadata: dict - Additional metadata
    
    Returns:
        Dictionary containing the agent's response and metadata
    
    Example:
        >>> result = run_agent({
        ...     "query": "What is the weather like today?",
        ...     "type": "question",
        ...     "context": {"user_id": "12345"}
        ... })
        >>> print(result["response"])
    """
    try:
        agent = get_agent_instance()
        return agent.run(input_dict)
    except Exception as e:
        logger.error(f"Critical error in run_agent: {e}")
        return {
            "success": False,
            "error": str(e),
            "response": f"Critical system error: {str(e)}",
            "action": "system_error",
            "confidence": 0.0,
            "timestamp": time.time()
        }


if __name__ == "__main__":
    # Example usage
    test_input = {
        "query": "Hello, can you help me understand what you can do?",
        "type": "introduction",
        "context": {"session": "test_session"}
    }
    
    result = run_agent(test_input)
    print("Agent Response:")
    print(json.dumps(result, indent=2))