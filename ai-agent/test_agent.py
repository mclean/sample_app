"""
Test harness for the Pinecone-powered AI Agent.
"""

import json
import time
import logging
from typing import Dict, Any, List
from main import run_agent, get_agent_instance

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentTester:
    """Test harness for the AI Agent."""
    
    def __init__(self):
        """Initialize the test harness."""
        self.test_results = []
        self.start_time = None
        logger.info("Initialized Agent Tester")
    
    def run_single_test(self, test_input: Dict[str, Any], expected_outcome: str = None) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Args:
            test_input: Input dictionary for the agent
            expected_outcome: Expected outcome description (optional)
            
        Returns:
            Dictionary containing test results
        """
        test_start = time.time()
        
        try:
            logger.info(f"Running test: {test_input.get('query', 'Unknown query')[:50]}...")
            
            # Run the agent
            result = run_agent(test_input)
            
            # Calculate test execution time
            test_time = time.time() - test_start
            
            # Evaluate the result
            test_result = {
                "input": test_input,
                "output": result,
                "success": result.get("success", False),
                "execution_time": test_time,
                "agent_execution_time": result.get("execution_time", 0),
                "response_length": len(result.get("response", "")),
                "memories_used": result.get("memories_used", 0),
                "confidence": result.get("confidence", 0),
                "expected_outcome": expected_outcome,
                "timestamp": time.time()
            }
            
            # Add evaluation metrics
            test_result["evaluation"] = self._evaluate_response(result, expected_outcome)
            
            self.test_results.append(test_result)
            
            logger.info(f"Test completed in {test_time:.2f}s - Success: {result.get('success', False)}")
            return test_result
            
        except Exception as e:
            test_time = time.time() - test_start
            error_result = {
                "input": test_input,
                "error": str(e),
                "success": False,
                "execution_time": test_time,
                "timestamp": time.time(),
                "evaluation": {"overall_score": 0, "error": str(e)}
            }
            
            self.test_results.append(error_result)
            logger.error(f"Test failed with error: {e}")
            return error_result
    
    def run_test_suite(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a suite of test cases.
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            Dictionary containing suite results
        """
        suite_start = time.time()
        self.start_time = suite_start
        
        logger.info(f"Running test suite with {len(test_cases)} test cases")
        
        suite_results = {
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "summary": {},
            "start_time": suite_start
        }
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Running test {i}/{len(test_cases)}")
            
            result = self.run_single_test(
                test_case.get("input"),
                test_case.get("expected_outcome")
            )
            
            suite_results["test_results"].append(result)
            
            if result.get("success", False):
                suite_results["passed_tests"] += 1
            else:
                suite_results["failed_tests"] += 1
        
        # Calculate suite summary
        suite_time = time.time() - suite_start
        suite_results["total_execution_time"] = suite_time
        suite_results["summary"] = self._generate_suite_summary(suite_results)
        
        logger.info(f"Test suite completed in {suite_time:.2f}s")
        logger.info(f"Results: {suite_results['passed_tests']}/{suite_results['total_tests']} tests passed")
        
        return suite_results
    
    def get_default_test_cases(self) -> List[Dict[str, Any]]:
        """Get a default set of test cases for the agent."""
        return [
            {
                "name": "Basic Greeting",
                "input": {
                    "query": "Hello! What can you help me with?",
                    "type": "greeting"
                },
                "expected_outcome": "Friendly greeting and explanation of capabilities"
            },
            {
                "name": "Information Request",
                "input": {
                    "query": "Can you explain what artificial intelligence is?",
                    "type": "question"
                },
                "expected_outcome": "Informative response about AI"
            },
            {
                "name": "Task Request",
                "input": {
                    "query": "Help me create a to-do list for organizing my workspace",
                    "type": "task"
                },
                "expected_outcome": "Structured list of workspace organization tasks"
            },
            {
                "name": "Follow-up Question",
                "input": {
                    "query": "What was my previous question about?",
                    "type": "followup"
                },
                "expected_outcome": "Reference to previous interaction using memory"
            },
            {
                "name": "Complex Reasoning",
                "input": {
                    "query": "If I have a team of 5 people and need to complete 20 tasks in 10 days, how should I organize the work?",
                    "type": "reasoning"
                },
                "expected_outcome": "Detailed work organization strategy"
            },
            {
                "name": "Edge Case - Empty Query",
                "input": {
                    "query": "",
                    "type": "edge_case"
                },
                "expected_outcome": "Error handling for empty input"
            },
            {
                "name": "Personal Preference",
                "input": {
                    "query": "I prefer brief, concise answers. Can you help me with project management?",
                    "type": "preference"
                },
                "expected_outcome": "Concise project management advice, storing preference"
            }
        ]
    
    def benchmark_performance(self, num_iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark agent performance with multiple iterations.
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            Performance benchmark results
        """
        logger.info(f"Running performance benchmark with {num_iterations} iterations")
        
        benchmark_input = {
            "query": "What are the key principles of effective communication?",
            "type": "benchmark"
        }
        
        execution_times = []
        success_count = 0
        confidence_scores = []
        memory_usage = []
        
        for i in range(num_iterations):
            result = self.run_single_test(benchmark_input)
            
            execution_times.append(result.get("agent_execution_time", 0))
            if result.get("success", False):
                success_count += 1
            confidence_scores.append(result.get("confidence", 0))
            memory_usage.append(result.get("memories_used", 0))
        
        # Calculate statistics
        avg_execution_time = sum(execution_times) / len(execution_times)
        min_execution_time = min(execution_times)
        max_execution_time = max(execution_times)
        success_rate = success_count / num_iterations
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        avg_memory_usage = sum(memory_usage) / len(memory_usage)
        
        benchmark_results = {
            "iterations": num_iterations,
            "performance_metrics": {
                "average_execution_time": avg_execution_time,
                "min_execution_time": min_execution_time,
                "max_execution_time": max_execution_time,
                "success_rate": success_rate,
                "average_confidence": avg_confidence,
                "average_memory_usage": avg_memory_usage
            },
            "raw_data": {
                "execution_times": execution_times,
                "confidence_scores": confidence_scores,
                "memory_usage": memory_usage
            }
        }
        
        logger.info(f"Benchmark completed - Avg time: {avg_execution_time:.2f}s, Success rate: {success_rate:.2%}")
        return benchmark_results
    
    def test_memory_functionality(self) -> Dict[str, Any]:
        """Test the memory storage and retrieval functionality."""
        logger.info("Testing memory functionality")
        
        # First interaction - store some information
        first_result = self.run_single_test({
            "query": "My name is Alice and I work as a software engineer at TechCorp",
            "type": "personal_info"
        })
        
        time.sleep(2)  # Allow time for memory storage
        
        # Second interaction - try to recall the information
        second_result = self.run_single_test({
            "query": "What do you remember about me?",
            "type": "memory_recall"
        })
        
        # Analyze if memory was used
        memory_test_result = {
            "first_interaction": first_result,
            "second_interaction": second_result,
            "memory_stored": first_result.get("success", False),
            "memory_recalled": second_result.get("memories_used", 0) > 0,
            "memory_content_present": "Alice" in second_result.get("output", {}).get("response", "").lower() or 
                                    "software engineer" in second_result.get("output", {}).get("response", "").lower()
        }
        
        logger.info(f"Memory test - Stored: {memory_test_result['memory_stored']}, "
                   f"Recalled: {memory_test_result['memory_recalled']}")
        
        return memory_test_result
    
    def _evaluate_response(self, result: Dict[str, Any], expected_outcome: str = None) -> Dict[str, Any]:
        """Evaluate the quality of an agent response."""
        evaluation = {
            "overall_score": 0,
            "criteria": {}
        }
        
        # Basic success check
        if result.get("success", False):
            evaluation["criteria"]["success"] = 1
        else:
            evaluation["criteria"]["success"] = 0
            evaluation["overall_score"] = 0
            return evaluation
        
        # Response length check (should have substantive content)
        response_length = len(result.get("response", ""))
        if response_length > 50:
            evaluation["criteria"]["response_length"] = 1
        elif response_length > 10:
            evaluation["criteria"]["response_length"] = 0.5
        else:
            evaluation["criteria"]["response_length"] = 0
        
        # Confidence check
        confidence = result.get("confidence", 0)
        evaluation["criteria"]["confidence"] = min(confidence, 1.0)
        
        # Execution time check (should be reasonable)
        exec_time = result.get("execution_time", 0)
        if exec_time < 5:
            evaluation["criteria"]["execution_time"] = 1
        elif exec_time < 10:
            evaluation["criteria"]["execution_time"] = 0.7
        else:
            evaluation["criteria"]["execution_time"] = 0.3
        
        # Memory usage check (should use memory when relevant)
        memories_used = result.get("memories_used", 0)
        evaluation["criteria"]["memory_usage"] = min(memories_used / 3.0, 1.0)  # Normalize to 0-1
        
        # Calculate overall score
        criteria_scores = list(evaluation["criteria"].values())
        evaluation["overall_score"] = sum(criteria_scores) / len(criteria_scores)
        
        return evaluation
    
    def _generate_suite_summary(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the test suite results."""
        test_results = suite_results["test_results"]
        
        if not test_results:
            return {"error": "No test results to summarize"}
        
        # Calculate averages
        avg_execution_time = sum(r.get("execution_time", 0) for r in test_results) / len(test_results)
        avg_confidence = sum(r.get("confidence", 0) for r in test_results) / len(test_results)
        avg_response_length = sum(r.get("response_length", 0) for r in test_results) / len(test_results)
        avg_memories_used = sum(r.get("memories_used", 0) for r in test_results) / len(test_results)
        
        # Calculate evaluation scores
        evaluation_scores = [r.get("evaluation", {}).get("overall_score", 0) for r in test_results]
        avg_evaluation_score = sum(evaluation_scores) / len(evaluation_scores)
        
        return {
            "pass_rate": suite_results["passed_tests"] / suite_results["total_tests"],
            "average_execution_time": avg_execution_time,
            "average_confidence": avg_confidence,
            "average_response_length": avg_response_length,
            "average_memories_used": avg_memories_used,
            "average_evaluation_score": avg_evaluation_score,
            "total_execution_time": suite_results["total_execution_time"]
        }
    
    def save_results(self, filename: str = "test_results.json"):
        """Save test results to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump({
                    "test_results": self.test_results,
                    "summary": self._generate_suite_summary({
                        "test_results": self.test_results,
                        "passed_tests": sum(1 for r in self.test_results if r.get("success", False)),
                        "total_tests": len(self.test_results),
                        "total_execution_time": sum(r.get("execution_time", 0) for r in self.test_results)
                    }),
                    "timestamp": time.time()
                }, f, indent=2)
            logger.info(f"Test results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving test results: {e}")


def main():
    """Main function to run tests."""
    print("🤖 AI Agent Test Harness")
    print("=" * 50)
    
    tester = AgentTester()
    
    # Run default test suite
    print("\n📋 Running Default Test Suite...")
    test_cases = tester.get_default_test_cases()
    suite_results = tester.run_test_suite(test_cases)
    
    # Print summary
    summary = suite_results["summary"]
    print(f"\n📊 Test Suite Summary:")
    print(f"  Tests Passed: {suite_results['passed_tests']}/{suite_results['total_tests']}")
    print(f"  Pass Rate: {summary.get('pass_rate', 0):.1%}")
    print(f"  Average Execution Time: {summary.get('average_execution_time', 0):.2f}s")
    print(f"  Average Confidence: {summary.get('average_confidence', 0):.2f}")
    print(f"  Average Evaluation Score: {summary.get('average_evaluation_score', 0):.2f}/1.0")
    
    # Test memory functionality
    print("\n🧠 Testing Memory Functionality...")
    memory_results = tester.test_memory_functionality()
    print(f"  Memory Storage: {'✅' if memory_results['memory_stored'] else '❌'}")
    print(f"  Memory Retrieval: {'✅' if memory_results['memory_recalled'] else '❌'}")
    print(f"  Content Recall: {'✅' if memory_results['memory_content_present'] else '❌'}")
    
    # Run performance benchmark
    print("\n⚡ Running Performance Benchmark...")
    benchmark_results = tester.benchmark_performance(5)
    metrics = benchmark_results["performance_metrics"]
    print(f"  Average Execution Time: {metrics['average_execution_time']:.2f}s")
    print(f"  Success Rate: {metrics['success_rate']:.1%}")
    print(f"  Average Confidence: {metrics['average_confidence']:.2f}")
    
    # Save results
    tester.save_results("agent_test_results.json")
    print(f"\n💾 Test results saved to agent_test_results.json")
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main()