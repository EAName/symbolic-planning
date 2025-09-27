"""
Evaluation Metrics and Benchmarking for PDDL-INSTRUCT Framework
Implements comprehensive evaluation metrics for symbolic planning tasks.
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
import statistics
from pddl_parser import PDDLDomain, PDDLProblem
from logical_reasoning import LogicalReasoner, ReasoningChain
from state_verification import StateVerifier, VerificationResult


class EvaluationMetric(Enum):
    """Types of evaluation metrics"""
    PLANNING_ACCURACY = "planning_accuracy"
    REASONING_ACCURACY = "reasoning_accuracy"
    EXECUTION_TIME = "execution_time"
    PLAN_LENGTH = "plan_length"
    REASONING_STEPS = "reasoning_steps"
    VERIFICATION_SUCCESS = "verification_success"


@dataclass
class EvaluationResult:
    """Results of an evaluation"""
    metric: EvaluationMetric
    value: float
    details: Dict[str, Any]
    timestamp: float


@dataclass
class BenchmarkResult:
    """Results of a benchmark evaluation"""
    domain_name: str
    problem_name: str
    results: List[EvaluationResult]
    overall_score: float
    success: bool


class PDDLEvaluator:
    """Evaluates PDDL planning and reasoning performance"""
    
    def __init__(self, domain: PDDLDomain):
        self.domain = domain
        self.reasoner = LogicalReasoner(domain)
        self.verifier = StateVerifier(domain)
        self.evaluation_history = []
    
    def evaluate_planning_accuracy(self, problem: PDDLProblem, 
                                 predicted_plan: List[str]) -> EvaluationResult:
        """Evaluate planning accuracy"""
        start_time = time.time()
        
        # Verify plan execution
        verification_report = self.verifier.verify_plan_execution(
            predicted_plan, problem.init_state, problem.goal_state
        )
        
        # Calculate accuracy
        accuracy = 1.0 if verification_report.result == VerificationResult.VALID else 0.0
        
        execution_time = time.time() - start_time
        
        details = {
            "plan": predicted_plan,
            "verification_result": verification_report.result.value,
            "verification_details": verification_report.details,
            "execution_time": execution_time,
            "errors": verification_report.errors,
            "warnings": verification_report.warnings
        }
        
        result = EvaluationResult(
            metric=EvaluationMetric.PLANNING_ACCURACY,
            value=accuracy,
            details=details,
            timestamp=time.time()
        )
        
        self.evaluation_history.append(result)
        return result
    
    def evaluate_reasoning_accuracy(self, problem: PDDLProblem, 
                                  predicted_plan: List[str]) -> EvaluationResult:
        """Evaluate reasoning accuracy by checking reasoning chains"""
        start_time = time.time()
        
        # Generate reasoning chain
        reasoning_chain = self.reasoner.generate_reasoning_chain(problem, predicted_plan)
        
        # Calculate reasoning accuracy based on successful reasoning steps
        total_steps = len(reasoning_chain.steps)
        successful_steps = sum(1 for step in reasoning_chain.steps if step.success)
        
        reasoning_accuracy = successful_steps / total_steps if total_steps > 0 else 0.0
        
        execution_time = time.time() - start_time
        
        details = {
            "reasoning_chain": reasoning_chain,
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "execution_time": execution_time,
            "explanation": reasoning_chain.explanation
        }
        
        result = EvaluationResult(
            metric=EvaluationMetric.REASONING_ACCURACY,
            value=reasoning_accuracy,
            details=details,
            timestamp=time.time()
        )
        
        self.evaluation_history.append(result)
        return result
    
    def evaluate_execution_time(self, problem: PDDLProblem, 
                              predicted_plan: List[str]) -> EvaluationResult:
        """Evaluate execution time"""
        start_time = time.time()
        
        # Execute the plan
        reasoning_chain = self.reasoner.generate_reasoning_chain(problem, predicted_plan)
        
        execution_time = time.time() - start_time
        
        details = {
            "plan": predicted_plan,
            "execution_time": execution_time,
            "success": reasoning_chain.success
        }
        
        result = EvaluationResult(
            metric=EvaluationMetric.EXECUTION_TIME,
            value=execution_time,
            details=details,
            timestamp=time.time()
        )
        
        self.evaluation_history.append(result)
        return result
    
    def evaluate_plan_length(self, predicted_plan: List[str]) -> EvaluationResult:
        """Evaluate plan length"""
        plan_length = len(predicted_plan)
        
        details = {
            "plan": predicted_plan,
            "length": plan_length
        }
        
        result = EvaluationResult(
            metric=EvaluationMetric.PLAN_LENGTH,
            value=plan_length,
            details=details,
            timestamp=time.time()
        )
        
        self.evaluation_history.append(result)
        return result
    
    def evaluate_reasoning_steps(self, problem: PDDLProblem, 
                               predicted_plan: List[str]) -> EvaluationResult:
        """Evaluate number of reasoning steps"""
        reasoning_chain = self.reasoner.generate_reasoning_chain(problem, predicted_plan)
        num_steps = len(reasoning_chain.steps)
        
        details = {
            "reasoning_chain": reasoning_chain,
            "num_steps": num_steps,
            "steps": [step.description for step in reasoning_chain.steps]
        }
        
        result = EvaluationResult(
            metric=EvaluationMetric.REASONING_STEPS,
            value=num_steps,
            details=details,
            timestamp=time.time()
        )
        
        self.evaluation_history.append(result)
        return result
    
    def evaluate_verification_success(self, problem: PDDLProblem, 
                                    predicted_plan: List[str]) -> EvaluationResult:
        """Evaluate verification success rate"""
        verification_report = self.verifier.verify_plan_execution(
            predicted_plan, problem.init_state, problem.goal_state
        )
        
        success_rate = 1.0 if verification_report.result == VerificationResult.VALID else 0.0
        
        details = {
            "verification_report": verification_report,
            "success": verification_report.result == VerificationResult.VALID,
            "errors": verification_report.errors,
            "warnings": verification_report.warnings
        }
        
        result = EvaluationResult(
            metric=EvaluationMetric.VERIFICATION_SUCCESS,
            value=success_rate,
            details=details,
            timestamp=time.time()
        )
        
        self.evaluation_history.append(result)
        return result
    
    def comprehensive_evaluation(self, problem: PDDLProblem, 
                               predicted_plan: List[str]) -> BenchmarkResult:
        """Perform comprehensive evaluation of a plan"""
        results = []
        
        # Evaluate all metrics
        results.append(self.evaluate_planning_accuracy(problem, predicted_plan))
        results.append(self.evaluate_reasoning_accuracy(problem, predicted_plan))
        results.append(self.evaluate_execution_time(problem, predicted_plan))
        results.append(self.evaluate_plan_length(predicted_plan))
        results.append(self.evaluate_reasoning_steps(problem, predicted_plan))
        results.append(self.evaluate_verification_success(problem, predicted_plan))
        
        # Calculate overall score (weighted average)
        weights = {
            EvaluationMetric.PLANNING_ACCURACY: 0.4,
            EvaluationMetric.REASONING_ACCURACY: 0.3,
            EvaluationMetric.VERIFICATION_SUCCESS: 0.2,
            EvaluationMetric.EXECUTION_TIME: 0.1
        }
        
        overall_score = 0.0
        for result in results:
            if result.metric in weights:
                # Normalize execution time (lower is better)
                if result.metric == EvaluationMetric.EXECUTION_TIME:
                    normalized_value = max(0, 1 - result.value / 10.0)  # Assume 10s is max
                else:
                    normalized_value = result.value
                
                overall_score += weights[result.metric] * normalized_value
        
        # Determine success
        planning_accuracy = next(r.value for r in results if r.metric == EvaluationMetric.PLANNING_ACCURACY)
        success = planning_accuracy > 0.8  # 80% threshold
        
        benchmark_result = BenchmarkResult(
            domain_name=problem.domain_name,
            problem_name=problem.name,
            results=results,
            overall_score=overall_score,
            success=success
        )
        
        return benchmark_result


class BenchmarkSuite:
    """Comprehensive benchmark suite for PDDL planning"""
    
    def __init__(self):
        self.benchmarks = []
        self.results = []
    
    def add_benchmark(self, domain: PDDLDomain, problem: PDDLProblem, 
                     predicted_plan: List[str], expected_plan: Optional[List[str]] = None):
        """Add a benchmark to the suite"""
        benchmark = {
            "domain": domain,
            "problem": problem,
            "predicted_plan": predicted_plan,
            "expected_plan": expected_plan
        }
        self.benchmarks.append(benchmark)
    
    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks in the suite"""
        results = []
        
        for benchmark in self.benchmarks:
            domain = benchmark["domain"]
            problem = benchmark["problem"]
            predicted_plan = benchmark["predicted_plan"]
            
            evaluator = PDDLEvaluator(domain)
            result = evaluator.comprehensive_evaluation(problem, predicted_plan)
            results.append(result)
        
        self.results = results
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report"""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Calculate aggregate statistics
        planning_accuracies = [r.results[0].value for r in self.results if r.results[0].metric == EvaluationMetric.PLANNING_ACCURACY]
        reasoning_accuracies = [r.results[1].value for r in self.results if r.results[1].metric == EvaluationMetric.REASONING_ACCURACY]
        execution_times = [r.results[2].value for r in self.results if r.results[2].metric == EvaluationMetric.EXECUTION_TIME]
        plan_lengths = [r.results[3].value for r in self.results if r.results[3].metric == EvaluationMetric.PLAN_LENGTH]
        reasoning_steps = [r.results[4].value for r in self.results if r.results[4].metric == EvaluationMetric.REASONING_STEPS]
        verification_successes = [r.results[5].value for r in self.results if r.results[5].metric == EvaluationMetric.VERIFICATION_SUCCESS]
        
        report = {
            "summary": {
                "total_benchmarks": len(self.results),
                "successful_benchmarks": sum(1 for r in self.results if r.success),
                "success_rate": sum(1 for r in self.results if r.success) / len(self.results),
                "average_overall_score": statistics.mean([r.overall_score for r in self.results])
            },
            "metrics": {
                "planning_accuracy": {
                    "mean": statistics.mean(planning_accuracies),
                    "std": statistics.stdev(planning_accuracies) if len(planning_accuracies) > 1 else 0,
                    "min": min(planning_accuracies),
                    "max": max(planning_accuracies)
                },
                "reasoning_accuracy": {
                    "mean": statistics.mean(reasoning_accuracies),
                    "std": statistics.stdev(reasoning_accuracies) if len(reasoning_accuracies) > 1 else 0,
                    "min": min(reasoning_accuracies),
                    "max": max(reasoning_accuracies)
                },
                "execution_time": {
                    "mean": statistics.mean(execution_times),
                    "std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    "min": min(execution_times),
                    "max": max(execution_times)
                },
                "plan_length": {
                    "mean": statistics.mean(plan_lengths),
                    "std": statistics.stdev(plan_lengths) if len(plan_lengths) > 1 else 0,
                    "min": min(plan_lengths),
                    "max": max(plan_lengths)
                },
                "reasoning_steps": {
                    "mean": statistics.mean(reasoning_steps),
                    "std": statistics.stdev(reasoning_steps) if len(reasoning_steps) > 1 else 0,
                    "min": min(reasoning_steps),
                    "max": max(reasoning_steps)
                },
                "verification_success": {
                    "mean": statistics.mean(verification_successes),
                    "std": statistics.stdev(verification_successes) if len(verification_successes) > 1 else 0,
                    "min": min(verification_successes),
                    "max": max(verification_successes)
                }
            },
            "detailed_results": [
                {
                    "domain": r.domain_name,
                    "problem": r.problem_name,
                    "overall_score": r.overall_score,
                    "success": r.success,
                    "planning_accuracy": r.results[0].value,
                    "reasoning_accuracy": r.results[1].value,
                    "execution_time": r.results[2].value,
                    "plan_length": r.results[3].value,
                    "reasoning_steps": r.results[4].value,
                    "verification_success": r.results[5].value
                }
                for r in self.results
            ]
        }
        
        return report
    
    def export_results(self, file_path: str):
        """Export benchmark results to JSON file"""
        report = self.generate_report()
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def print_report(self):
        """Print a formatted benchmark report"""
        report = self.generate_report()
        
        print("=" * 80)
        print("PDDL-INSTRUCT BENCHMARK REPORT")
        print("=" * 80)
        
        print(f"Total Benchmarks: {report['summary']['total_benchmarks']}")
        print(f"Successful Benchmarks: {report['summary']['successful_benchmarks']}")
        print(f"Success Rate: {report['summary']['success_rate']:.2%}")
        print(f"Average Overall Score: {report['summary']['average_overall_score']:.3f}")
        print()
        
        print("METRIC STATISTICS:")
        print("-" * 40)
        
        for metric_name, stats in report['metrics'].items():
            print(f"{metric_name.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Std:  {stats['std']:.3f}")
            print(f"  Min:  {stats['min']:.3f}")
            print(f"  Max:  {stats['max']:.3f}")
            print()
        
        print("DETAILED RESULTS:")
        print("-" * 40)
        
        for result in report['detailed_results']:
            print(f"Domain: {result['domain']}, Problem: {result['problem']}")
            print(f"  Overall Score: {result['overall_score']:.3f}")
            print(f"  Success: {result['success']}")
            print(f"  Planning Accuracy: {result['planning_accuracy']:.3f}")
            print(f"  Reasoning Accuracy: {result['reasoning_accuracy']:.3f}")
            print(f"  Execution Time: {result['execution_time']:.3f}s")
            print(f"  Plan Length: {result['plan_length']}")
            print(f"  Reasoning Steps: {result['reasoning_steps']}")
            print(f"  Verification Success: {result['verification_success']:.3f}")
            print()


# Example usage
if __name__ == "__main__":
    from pddl_parser import PDDLParser, load_pddl_file
    
    # Load a domain and problem for testing
    parser = PDDLParser()
    domain_text = load_pddl_file("examples/blocksworld_domain.pddl")
    problem_text = load_pddl_file("examples/blocksworld_problem.pddl")
    
    domain = parser.parse_domain(domain_text)
    problem = parser.parse_problem(problem_text)
    
    # Create evaluator
    evaluator = PDDLEvaluator(domain)
    
    # Test with a sample plan
    sample_plan = [
        "pickup(a)",
        "stack(a, b)",
        "pickup(c)",
        "stack(c, a)"
    ]
    
    # Run comprehensive evaluation
    result = evaluator.comprehensive_evaluation(problem, sample_plan)
    
    print("Evaluation Result:")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Success: {result.success}")
    print()
    
    for eval_result in result.results:
        print(f"{eval_result.metric.value}: {eval_result.value:.3f}")
    
    # Create benchmark suite
    benchmark_suite = BenchmarkSuite()
    benchmark_suite.add_benchmark(domain, problem, sample_plan)
    
    # Run benchmarks
    benchmark_results = benchmark_suite.run_benchmarks()
    
    # Generate and print report
    benchmark_suite.print_report()
