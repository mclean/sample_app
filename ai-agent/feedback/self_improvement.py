"""
Self-improvement module for summarizing interactions and storing learnings.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.openai_api import OpenAIClient
from utils.logger import JSONLLogger

logger = logging.getLogger(__name__)


class SelfImprovementEngine:
    """Engine for summarizing interactions and learning from feedback."""
    
    def __init__(self, config: Dict[str, Any], json_logger: JSONLLogger):
        """
        Initialize the self-improvement engine.
        
        Args:
            config: Configuration dictionary
            json_logger: Logger instance for structured logging
        """
        self.config = config
        self.json_logger = json_logger
        
        # Initialize OpenAI client for summarization
        self.openai_client = OpenAIClient(
            model=config.get("reasoning_settings", {}).get("model", "gpt-4")
        )
        
        logger.info("Initialized self-improvement engine")
    
    def summarize_interaction(
        self,
        user_input: Dict[str, Any],
        agent_response: Dict[str, Any],
        memories_used: List[str],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Summarize an interaction for storage as a memory.
        
        Args:
            user_input: The user's input and metadata
            agent_response: The agent's response and metadata
            memories_used: List of memories that were retrieved
            execution_context: Context about the execution
            
        Returns:
            Dictionary containing the interaction summary
        """
        try:
            # Create prompt for summarization
            summary_prompt = self._build_summarization_prompt(
                user_input, agent_response, memories_used, execution_context
            )
            
            # Generate summary using GPT-4
            response = self.openai_client.chat_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                system_message="You are an interaction summarization system. Create concise, informative summaries.",
                temperature=0.3,
                max_tokens=500
            )
            
            # Structure the summary
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "interaction_type": user_input.get("type", "general"),
                "user_query": user_input.get("query", ""),
                "agent_action": agent_response.get("action", "respond"),
                "summary": response["content"],
                "success": agent_response.get("success", True),
                "confidence": agent_response.get("confidence", 0.8),
                "memories_referenced": len(memories_used),
                "execution_time": execution_context.get("execution_time", 0),
                "tokens_used": response.get("usage", {}).get("total_tokens", 0),
                "key_learnings": self._extract_learnings(response["content"]),
                "session_id": execution_context.get("session_id", "unknown")
            }
            
            logger.info("Generated interaction summary")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing interaction: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "summary": f"Interaction on {datetime.utcnow().isoformat()}: Error in summarization",
                "success": False
            }
    
    def analyze_performance_trends(
        self,
        recent_interactions: List[Dict[str, Any]],
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Analyze performance trends from recent interactions.
        
        Args:
            recent_interactions: List of recent interaction logs
            time_window_hours: Time window for analysis in hours
            
        Returns:
            Dictionary containing performance analysis
        """
        try:
            if not recent_interactions:
                return {"status": "no_data", "message": "No recent interactions to analyze"}
            
            # Calculate metrics
            total_interactions = len(recent_interactions)
            successful_interactions = sum(1 for i in recent_interactions if i.get("success", True))
            success_rate = successful_interactions / total_interactions if total_interactions > 0 else 0
            
            # Calculate average confidence and execution time
            confidences = [i.get("confidence", 0.8) for i in recent_interactions if "confidence" in i]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8
            
            execution_times = [i.get("execution_time", 0) for i in recent_interactions if "execution_time" in i]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            
            # Analyze interaction types
            interaction_types = {}
            for interaction in recent_interactions:
                itype = interaction.get("interaction_type", "unknown")
                interaction_types[itype] = interaction_types.get(itype, 0) + 1
            
            # Identify common issues
            errors = [i.get("error", "") for i in recent_interactions if i.get("error")]
            common_issues = self._identify_common_issues(errors)
            
            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(
                success_rate, avg_confidence, avg_execution_time, interaction_types, common_issues
            )
            
            analysis = {
                "timestamp": datetime.utcnow().isoformat(),
                "time_window_hours": time_window_hours,
                "metrics": {
                    "total_interactions": total_interactions,
                    "success_rate": success_rate,
                    "average_confidence": avg_confidence,
                    "average_execution_time": avg_execution_time
                },
                "interaction_breakdown": interaction_types,
                "common_issues": common_issues,
                "improvement_suggestions": suggestions,
                "status": "completed"
            }
            
            logger.info(f"Analyzed performance trends for {total_interactions} interactions")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def generate_learning_insights(
        self,
        interaction_summaries: List[str],
        memory_usage_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate learning insights from interaction patterns.
        
        Args:
            interaction_summaries: List of interaction summaries
            memory_usage_patterns: Patterns in memory usage
            
        Returns:
            Dictionary containing learning insights
        """
        try:
            # Create prompt for insight generation
            insights_prompt = f"""
Analyze these interaction summaries and memory usage patterns to identify learning opportunities:

Interaction Summaries:
{chr(10).join([f"- {summary}" for summary in interaction_summaries[:10]])}

Memory Usage Patterns:
{memory_usage_patterns}

Provide insights about:
1. Common user needs and preferences
2. Areas where the agent performs well
3. Areas needing improvement
4. Patterns in successful vs unsuccessful interactions
5. Recommendations for better memory organization

Format as structured analysis with clear sections.
"""
            
            response = self.openai_client.chat_completion(
                messages=[{"role": "user", "content": insights_prompt}],
                system_message="You are a learning analysis system. Identify patterns and improvement opportunities.",
                temperature=0.5,
                max_tokens=800
            )
            
            insights = {
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": response["content"],
                "data_points_analyzed": len(interaction_summaries),
                "memory_patterns": memory_usage_patterns,
                "tokens_used": response.get("usage", {}).get("total_tokens", 0),
                "key_recommendations": self._extract_recommendations(response["content"])
            }
            
            logger.info("Generated learning insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating learning insights: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "analysis": "Unable to generate insights due to error"
            }
    
    def create_memory_summary(
        self,
        interaction_summary: Dict[str, Any],
        memory_type: str = "interaction"
    ) -> str:
        """
        Create a memory-optimized summary for storage in Pinecone.
        
        Args:
            interaction_summary: Summary of the interaction
            memory_type: Type of memory being created
            
        Returns:
            String formatted for memory storage
        """
        try:
            user_query = interaction_summary.get("user_query", "")
            agent_action = interaction_summary.get("agent_action", "respond")
            summary_text = interaction_summary.get("summary", "")
            timestamp = interaction_summary.get("timestamp", "")
            
            # Create structured memory text
            memory_text = f"""
Interaction on {timestamp[:10]}:
User Query: {user_query}
Agent Action: {agent_action}
Summary: {summary_text}
Success: {interaction_summary.get('success', True)}
Confidence: {interaction_summary.get('confidence', 0.8)}
Key Learnings: {'; '.join(interaction_summary.get('key_learnings', []))}
""".strip()
            
            return memory_text
            
        except Exception as e:
            logger.error(f"Error creating memory summary: {e}")
            return f"Interaction summary from {datetime.utcnow().isoformat()}: {str(e)}"
    
    def _build_summarization_prompt(
        self,
        user_input: Dict[str, Any],
        agent_response: Dict[str, Any],
        memories_used: List[str],
        execution_context: Dict[str, Any]
    ) -> str:
        """Build prompt for interaction summarization."""
        return f"""
Summarize this interaction between a user and an AI agent:

User Input:
- Query: {user_input.get('query', '')}
- Type: {user_input.get('type', 'general')}

Agent Response:
- Action: {agent_response.get('action', 'respond')}
- Success: {agent_response.get('success', True)}
- Confidence: {agent_response.get('confidence', 0.8)}
- Response: {agent_response.get('response', '')[:200]}...

Memories Used: {len(memories_used)} relevant memories

Execution Context:
- Time: {execution_context.get('execution_time', 0)} seconds
- Tokens: {execution_context.get('tokens_used', 0)}

Create a concise summary that captures:
1. What the user asked
2. How the agent responded
3. Whether it was successful
4. Any key insights or learnings
5. Areas for improvement

Keep the summary under 200 words and focus on actionable insights.
"""
    
    def _extract_learnings(self, summary_text: str) -> List[str]:
        """Extract key learnings from summary text."""
        learnings = []
        lines = summary_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['learning', 'insight', 'improvement', 'note']):
                if line.startswith('- ') or line.startswith('* '):
                    learnings.append(line[2:].strip())
                elif ':' in line:
                    learnings.append(line.split(':', 1)[1].strip())
        
        return learnings[:5]  # Limit to top 5 learnings
    
    def _identify_common_issues(self, errors: List[str]) -> List[str]:
        """Identify common issues from error messages."""
        if not errors:
            return []
        
        # Simple pattern matching for common issues
        issue_patterns = {
            "API": ["api", "key", "authentication", "rate limit"],
            "Memory": ["memory", "pinecone", "embedding", "retrieval"],
            "Processing": ["timeout", "processing", "parsing", "format"],
            "Input": ["input", "query", "malformed", "invalid"]
        }
        
        issues = []
        for error in errors:
            error_lower = error.lower()
            for issue_type, patterns in issue_patterns.items():
                if any(pattern in error_lower for pattern in patterns):
                    issues.append(f"{issue_type} related issues")
                    break
        
        return list(set(issues))
    
    def _generate_improvement_suggestions(
        self,
        success_rate: float,
        avg_confidence: float,
        avg_execution_time: float,
        interaction_types: Dict[str, int],
        common_issues: List[str]
    ) -> List[str]:
        """Generate improvement suggestions based on metrics."""
        suggestions = []
        
        if success_rate < 0.9:
            suggestions.append("Focus on improving error handling and edge cases")
        
        if avg_confidence < 0.7:
            suggestions.append("Enhance reasoning confidence through better context analysis")
        
        if avg_execution_time > 5.0:
            suggestions.append("Optimize response generation for faster execution")
        
        if common_issues:
            suggestions.append(f"Address common issues: {', '.join(common_issues)}")
        
        if len(interaction_types) == 1:
            suggestions.append("Expand capabilities to handle more diverse interaction types")
        
        if not suggestions:
            suggestions.append("Performance is good - continue monitoring for optimization opportunities")
        
        return suggestions
    
    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """Extract recommendations from analysis text."""
        recommendations = []
        lines = analysis_text.split('\n')
        
        in_recommendations = False
        for line in lines:
            line = line.strip()
            if 'recommendation' in line.lower() or 'suggest' in line.lower():
                in_recommendations = True
                continue
            
            if in_recommendations and (line.startswith('- ') or line.startswith('* ')):
                recommendations.append(line[2:].strip())
            elif in_recommendations and line.startswith(('1.', '2.', '3.', '4.', '5.')):
                recommendations.append(line[2:].strip())
        
        return recommendations[:5]  # Limit to top 5 recommendations