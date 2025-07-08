"""
Reasoning engine for GPT-4 prompt logic and decision making.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from utils.openai_api import OpenAIClient
from utils.logger import JSONLLogger

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """Engine for handling reasoning, decision making, and GPT-4 interactions."""
    
    def __init__(self, config: Dict[str, Any], json_logger: JSONLLogger):
        """
        Initialize the reasoning engine.
        
        Args:
            config: Configuration dictionary from agent_config.json
            json_logger: Logger instance for structured logging
        """
        self.config = config
        self.json_logger = json_logger
        
        # Initialize OpenAI client
        reasoning_config = config.get("reasoning_settings", {})
        self.openai_client = OpenAIClient(
            model=reasoning_config.get("model", "gpt-4")
        )
        
        # Store reasoning parameters
        self.temperature = reasoning_config.get("temperature", 0.7)
        self.max_tokens = reasoning_config.get("max_tokens", 1000)
        
        logger.info("Initialized reasoning engine")
    
    def decide_next_action(
        self,
        user_input: Dict[str, Any],
        relevant_memories: List[Tuple[str, float, Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decide the next action based on user input and context.
        
        Args:
            user_input: Dictionary containing user input and metadata
            relevant_memories: List of relevant memories from Pinecone
            context: Additional context for decision making
            
        Returns:
            Dictionary containing the decision and reasoning
        """
        try:
            start_time = datetime.utcnow()
            
            # Extract user query
            user_query = user_input.get("query", "")
            task_type = user_input.get("type", "general")
            
            # Prepare memory context
            memory_context = self._format_memories(relevant_memories)
            
            # Build agent context
            agent_context = {
                "name": self.config.get("agent_name", "AI Agent"),
                "description": self.config.get("agent_description", ""),
                "goals": self.config.get("goals", [])
            }
            
            # Generate reasoning response
            reasoning_response = self.openai_client.reasoning_prompt(
                user_input=user_query,
                relevant_memories=[mem[0] for mem in relevant_memories],
                agent_context=agent_context,
                task_type=task_type
            )
            
            # Parse and structure the response
            structured_response = self._structure_response(
                reasoning_response,
                user_input,
                memory_context
            )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Log the decision
            self.json_logger.log_decision(
                decision_type="reasoning",
                context={
                    "user_input": user_input,
                    "memories_count": len(relevant_memories),
                    "task_type": task_type
                },
                decision=structured_response.get("action", "respond"),
                confidence=structured_response.get("confidence", 0.8),
                alternatives=structured_response.get("alternatives", [])
            )
            
            # Add execution metadata
            structured_response.update({
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                "model_used": reasoning_response.get("model", "gpt-4"),
                "tokens_used": reasoning_response.get("usage", {})
            })
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": "error_response",
                "response": "I encountered an error while processing your request. Please try again."
            }
    
    def analyze_user_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Analyze user intent and categorize the input.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Dictionary containing intent analysis
        """
        try:
            system_message = """You are an intent analysis system. Analyze the user's input and provide:
1. Primary intent (question, request, task, conversation, etc.)
2. Urgency level (low, medium, high)
3. Required capabilities (memory_retrieval, reasoning, task_execution, etc.)
4. Emotional tone (neutral, positive, negative, frustrated, etc.)
5. Complexity level (simple, moderate, complex)

Format your response as JSON with these fields: intent, urgency, capabilities, tone, complexity, confidence."""

            messages = [{"role": "user", "content": user_input}]
            
            response = self.openai_client.chat_completion(
                messages=messages,
                system_message=system_message,
                temperature=0.3,
                max_tokens=300
            )
            
            # Try to parse JSON response
            try:
                intent_data = json.loads(response["content"])
                return intent_data
            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON
                return {
                    "intent": "general_query",
                    "urgency": "medium",
                    "capabilities": ["reasoning"],
                    "tone": "neutral",
                    "complexity": "moderate",
                    "confidence": 0.5,
                    "raw_response": response["content"]
                }
                
        except Exception as e:
            logger.error(f"Error analyzing user intent: {e}")
            return {
                "intent": "unknown",
                "urgency": "medium",
                "capabilities": ["reasoning"],
                "tone": "neutral",
                "complexity": "moderate",
                "confidence": 0.3,
                "error": str(e)
            }
    
    def generate_response_strategies(
        self,
        user_input: str,
        intent: Dict[str, Any],
        memories: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple response strategies for a given input.
        
        Args:
            user_input: The user's input
            intent: Intent analysis results
            memories: Relevant memories
            
        Returns:
            List of response strategies
        """
        try:
            memory_context = "\n".join([f"- {mem}" for mem in memories])
            
            system_message = f"""Based on the user input and intent analysis, generate 3 different response strategies.

Intent Analysis: {json.dumps(intent, indent=2)}

Relevant Memories:
{memory_context if memory_context else "No relevant memories"}

For each strategy, provide:
1. Approach (how to handle the request)
2. Key points to address
3. Tone and style
4. Estimated effectiveness (1-10)
5. Required resources

Format as JSON array of strategy objects."""

            messages = [{"role": "user", "content": user_input}]
            
            response = self.openai_client.chat_completion(
                messages=messages,
                system_message=system_message,
                temperature=0.8,
                max_tokens=800
            )
            
            try:
                strategies = json.loads(response["content"])
                return strategies if isinstance(strategies, list) else [strategies]
            except json.JSONDecodeError:
                # Fallback strategy
                return [{
                    "approach": "direct_response",
                    "key_points": ["Address user query directly"],
                    "tone": "helpful and informative",
                    "effectiveness": 7,
                    "resources": ["reasoning"],
                    "raw_response": response["content"]
                }]
                
        except Exception as e:
            logger.error(f"Error generating response strategies: {e}")
            return [{
                "approach": "error_handling",
                "key_points": ["Acknowledge error", "Provide fallback response"],
                "tone": "apologetic and helpful",
                "effectiveness": 5,
                "resources": ["basic_response"],
                "error": str(e)
            }]
    
    def _format_memories(self, memories: List[Tuple[str, float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Format memories for use in reasoning context."""
        formatted = []
        for content, score, metadata in memories:
            formatted.append({
                "content": content,
                "relevance_score": score,
                "memory_type": metadata.get("memory_type", "unknown"),
                "timestamp": metadata.get("timestamp", "unknown")
            })
        return formatted
    
    def _structure_response(
        self,
        reasoning_response: Dict[str, Any],
        user_input: Dict[str, Any],
        memory_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Structure the reasoning response into a standard format."""
        try:
            content = reasoning_response.get("content", "")
            
            # Try to extract structured information from the response
            lines = content.split('\n')
            
            # Default structured response
            structured = {
                "success": True,
                "action": "respond",
                "response": content,
                "confidence": 0.8,
                "reasoning_steps": [],
                "key_insights": [],
                "follow_up_questions": [],
                "alternatives": []
            }
            
            # Parse the response for structured elements
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Identify sections
                if "understanding" in line.lower() or "analysis" in line.lower():
                    current_section = "understanding"
                elif "reasoning" in line.lower() or "process" in line.lower():
                    current_section = "reasoning"
                elif "response" in line.lower() or "action" in line.lower():
                    current_section = "response"
                elif "follow-up" in line.lower() or "clarification" in line.lower():
                    current_section = "follow_up"
                
                # Extract content based on section
                if line.startswith('- ') or line.startswith('* '):
                    item = line[2:].strip()
                    if current_section == "reasoning":
                        structured["reasoning_steps"].append(item)
                    elif current_section == "follow_up":
                        structured["follow_up_questions"].append(item)
                    else:
                        structured["key_insights"].append(item)
            
            # Determine action type based on content
            if any(word in content.lower() for word in ["question", "clarify", "need more"]):
                structured["action"] = "request_clarification"
            elif any(word in content.lower() for word in ["task", "action", "do", "execute"]):
                structured["action"] = "execute_task"
            else:
                structured["action"] = "respond"
            
            # Add memory context information
            structured["memory_context"] = {
                "memories_used": len(memory_context),
                "memory_summary": [mem["content"][:100] + "..." for mem in memory_context[:3]]
            }
            
            return structured
            
        except Exception as e:
            logger.error(f"Error structuring response: {e}")
            return {
                "success": False,
                "action": "respond",
                "response": reasoning_response.get("content", "I apologize, but I had trouble processing your request."),
                "error": str(e),
                "confidence": 0.5
            }