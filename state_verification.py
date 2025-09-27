"""
State Transition and Action Applicability Verification
Implements rigorous verification of state transitions and action applicability for PDDL-INSTRUCT.
"""

from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from pddl_parser import PDDLDomain, PDDLAction
from logical_reasoning import LogicalReasoner, ReasoningStep, ReasoningStepType


class VerificationResult(Enum):
    """Results of verification operations"""
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class VerificationReport:
    """Comprehensive verification report"""
    result: VerificationResult
    details: List[str]
    errors: List[str]
    warnings: List[str]
    reasoning_steps: List[ReasoningStep]
    execution_trace: List[Dict[str, Any]]


class StateVerifier:
    """Verifies state transitions and action applicability"""
    
    def __init__(self, domain: PDDLDomain):
        self.domain = domain
        self.reasoner = LogicalReasoner(domain)
        self.verification_history = []
    
    def verify_action_applicability(self, action_name: str, parameters: Dict[str, str], 
                                  state: Set[str]) -> VerificationReport:
        """Verify if an action is applicable in a given state"""
        details = []
        errors = []
        warnings = []
        reasoning_steps = []
        execution_trace = []
        
        try:
            # Check if action exists in domain
            if action_name not in self.domain.actions:
                errors.append(f"Action '{action_name}' not found in domain")
                return VerificationReport(
                    result=VerificationResult.ERROR,
                    details=details,
                    errors=errors,
                    warnings=warnings,
                    reasoning_steps=reasoning_steps,
                    execution_trace=execution_trace
                )
            
            action = self.domain.actions[action_name]
            details.append(f"Action '{action_name}' found in domain")
            
            # Check parameter types
            param_verification = self._verify_parameters(action, parameters)
            if param_verification.result == VerificationResult.ERROR:
                errors.extend(param_verification.errors)
                return VerificationReport(
                    result=VerificationResult.ERROR,
                    details=details,
                    errors=errors,
                    warnings=warnings,
                    reasoning_steps=reasoning_steps,
                    execution_trace=execution_trace
                )
            
            details.extend(param_verification.details)
            warnings.extend(param_verification.warnings)
            
            # Check preconditions
            prec_verification = self._verify_preconditions(action, parameters, state)
            details.extend(prec_verification.details)
            errors.extend(prec_verification.errors)
            warnings.extend(prec_verification.warnings)
            reasoning_steps.extend(prec_verification.reasoning_steps)
            
            if prec_verification.result == VerificationResult.INVALID:
                return VerificationReport(
                    result=VerificationResult.INVALID,
                    details=details,
                    errors=errors,
                    warnings=warnings,
                    reasoning_steps=reasoning_steps,
                    execution_trace=execution_trace
                )
            
            # Action is applicable
            details.append(f"Action '{action_name}' is applicable in the given state")
            
            return VerificationReport(
                result=VerificationResult.VALID,
                details=details,
                errors=errors,
                warnings=warnings,
                reasoning_steps=reasoning_steps,
                execution_trace=execution_trace
            )
            
        except Exception as e:
            errors.append(f"Unexpected error during verification: {str(e)}")
            return VerificationReport(
                result=VerificationResult.ERROR,
                details=details,
                errors=errors,
                warnings=warnings,
                reasoning_steps=reasoning_steps,
                execution_trace=execution_trace
            )
    
    def verify_state_transition(self, action_name: str, parameters: Dict[str, str], 
                              initial_state: Set[str], expected_final_state: Set[str]) -> VerificationReport:
        """Verify if a state transition is correct"""
        details = []
        errors = []
        warnings = []
        reasoning_steps = []
        execution_trace = []
        
        try:
            # First verify action applicability
            applicability_report = self.verify_action_applicability(action_name, parameters, initial_state)
            
            if applicability_report.result == VerificationResult.INVALID:
                errors.append("Action is not applicable in initial state")
                errors.extend(applicability_report.errors)
                return VerificationReport(
                    result=VerificationResult.INVALID,
                    details=details,
                    errors=errors,
                    warnings=warnings,
                    reasoning_steps=reasoning_steps,
                    execution_trace=execution_trace
                )
            
            details.extend(applicability_report.details)
            warnings.extend(applicability_report.warnings)
            reasoning_steps.extend(applicability_report.reasoning_steps)
            
            # Apply action to get actual final state
            reasoning_chain = self.reasoner.reason_about_action_applicability(action_name, parameters, initial_state)
            actual_final_state = reasoning_chain.final_state
            
            # Compare actual vs expected final state
            state_comparison = self._compare_states(actual_final_state, expected_final_state)
            
            details.extend(state_comparison.details)
            errors.extend(state_comparison.errors)
            warnings.extend(state_comparison.warnings)
            
            if state_comparison.result == VerificationResult.VALID:
                details.append("State transition is correct")
                result = VerificationResult.VALID
            else:
                details.append("State transition is incorrect")
                result = VerificationResult.INVALID
            
            return VerificationReport(
                result=result,
                details=details,
                errors=errors,
                warnings=warnings,
                reasoning_steps=reasoning_steps,
                execution_trace=execution_trace
            )
            
        except Exception as e:
            errors.append(f"Unexpected error during state transition verification: {str(e)}")
            return VerificationReport(
                result=VerificationResult.ERROR,
                details=details,
                errors=errors,
                warnings=warnings,
                reasoning_steps=reasoning_steps,
                execution_trace=execution_trace
            )
    
    def verify_plan_execution(self, plan: List[str], initial_state: Set[str], 
                            goal_state: Set[str]) -> VerificationReport:
        """Verify if a plan correctly achieves the goal"""
        details = []
        errors = []
        warnings = []
        reasoning_steps = []
        execution_trace = []
        
        try:
            current_state = initial_state.copy()
            details.append(f"Starting plan execution from initial state: {', '.join(sorted(initial_state))}")
            
            for i, action_str in enumerate(plan):
                details.append(f"Executing action {i+1}: {action_str}")
                
                # Parse action string
                try:
                    action_name, parameters = self._parse_action_string(action_str)
                except ValueError as e:
                    errors.append(f"Error parsing action {i+1}: {str(e)}")
                    return VerificationReport(
                        result=VerificationResult.ERROR,
                        details=details,
                        errors=errors,
                        warnings=warnings,
                        reasoning_steps=reasoning_steps,
                        execution_trace=execution_trace
                    )
                
                # Verify action applicability
                applicability_report = self.verify_action_applicability(action_name, parameters, current_state)
                
                if applicability_report.result == VerificationResult.INVALID:
                    errors.append(f"Action {i+1} is not applicable in current state")
                    errors.extend(applicability_report.errors)
                    return VerificationReport(
                        result=VerificationResult.INVALID,
                        details=details,
                        errors=errors,
                        warnings=warnings,
                        reasoning_steps=reasoning_steps,
                        execution_trace=execution_trace
                    )
                
                # Apply action
                reasoning_chain = self.reasoner.reason_about_action_applicability(action_name, parameters, current_state)
                current_state = reasoning_chain.final_state
                
                details.append(f"Action {i+1} applied successfully. New state: {', '.join(sorted(current_state))}")
                reasoning_steps.extend(reasoning_chain.steps)
                
                execution_trace.append({
                    "action": action_str,
                    "state_before": current_state.copy(),
                    "state_after": current_state.copy(),
                    "success": True
                })
            
            # Check goal satisfaction
            goal_verification = self._verify_goal_satisfaction(current_state, goal_state)
            details.extend(goal_verification.details)
            errors.extend(goal_verification.errors)
            warnings.extend(goal_verification.warnings)
            
            if goal_verification.result == VerificationResult.VALID:
                details.append("Plan successfully achieves the goal")
                result = VerificationResult.VALID
            else:
                details.append("Plan does not achieve the goal")
                result = VerificationResult.INVALID
            
            return VerificationReport(
                result=result,
                details=details,
                errors=errors,
                warnings=warnings,
                reasoning_steps=reasoning_steps,
                execution_trace=execution_trace
            )
            
        except Exception as e:
            errors.append(f"Unexpected error during plan verification: {str(e)}")
            return VerificationReport(
                result=VerificationResult.ERROR,
                details=details,
                errors=errors,
                warnings=warnings,
                reasoning_steps=reasoning_steps,
                execution_trace=execution_trace
            )
    
    def _verify_parameters(self, action: PDDLAction, parameters: Dict[str, str]) -> VerificationReport:
        """Verify action parameters"""
        details = []
        errors = []
        warnings = []
        
        # Check if all required parameters are provided
        required_params = {param[0] for param in action.parameters}
        provided_params = set(parameters.keys())
        
        missing_params = required_params - provided_params
        if missing_params:
            errors.append(f"Missing parameters: {', '.join(missing_params)}")
        
        extra_params = provided_params - required_params
        if extra_params:
            warnings.append(f"Extra parameters provided: {', '.join(extra_params)}")
        
        # Check parameter types (if we have type information)
        for param_name, param_type in action.parameters:
            if param_name in parameters:
                # This is a simplified type check - in practice, you'd need more sophisticated type checking
                details.append(f"Parameter '{param_name}' has value '{parameters[param_name]}' of type '{param_type}'")
        
        if not errors:
            details.append("All parameters are valid")
            result = VerificationResult.VALID
        else:
            result = VerificationResult.INVALID
        
        return VerificationReport(
            result=result,
            details=details,
            errors=errors,
            warnings=warnings,
            reasoning_steps=[],
            execution_trace=[]
        )
    
    def _verify_preconditions(self, action: PDDLAction, parameters: Dict[str, str], 
                            state: Set[str]) -> VerificationReport:
        """Verify action preconditions"""
        details = []
        errors = []
        warnings = []
        reasoning_steps = []
        
        for precondition in action.preconditions:
            # Instantiate precondition with parameters
            instantiated_precond = self._instantiate_literal(precondition, parameters)
            details.append(f"Checking precondition: {instantiated_precond}")
            
            # Check if precondition is satisfied
            if not self._is_literal_satisfied(instantiated_precond, state):
                errors.append(f"Precondition not satisfied: {instantiated_precond}")
                
                # Create reasoning step
                step = ReasoningStep(
                    step_type=ReasoningStepType.PRECONDITION_CHECK,
                    description=f"Check precondition: {instantiated_precond}",
                    input_state=state.copy(),
                    output_state=state.copy(),
                    action=action.name,
                    parameters=parameters,
                    success=False,
                    explanation=f"Precondition {instantiated_precond} not satisfied in current state"
                )
                reasoning_steps.append(step)
            else:
                details.append(f"Precondition satisfied: {instantiated_precond}")
                
                # Create reasoning step
                step = ReasoningStep(
                    step_type=ReasoningStepType.PRECONDITION_CHECK,
                    description=f"Check precondition: {instantiated_precond}",
                    input_state=state.copy(),
                    output_state=state.copy(),
                    action=action.name,
                    parameters=parameters,
                    success=True,
                    explanation=f"Precondition {instantiated_precond} satisfied in current state"
                )
                reasoning_steps.append(step)
        
        if not errors:
            details.append("All preconditions are satisfied")
            result = VerificationResult.VALID
        else:
            result = VerificationResult.INVALID
        
        return VerificationReport(
            result=result,
            details=details,
            errors=errors,
            warnings=warnings,
            reasoning_steps=reasoning_steps,
            execution_trace=[]
        )
    
    def _compare_states(self, actual_state: Set[str], expected_state: Set[str]) -> VerificationReport:
        """Compare two states"""
        details = []
        errors = []
        warnings = []
        
        # Find differences
        missing_literals = expected_state - actual_state
        extra_literals = actual_state - expected_state
        
        if missing_literals:
            errors.append(f"Missing literals in actual state: {', '.join(missing_literals)}")
        
        if extra_literals:
            warnings.append(f"Extra literals in actual state: {', '.join(extra_literals)}")
        
        if not missing_literals and not extra_literals:
            details.append("States are identical")
            result = VerificationResult.VALID
        elif not missing_literals:
            details.append("States match (with extra literals)")
            result = VerificationResult.PARTIAL
        else:
            details.append("States do not match")
            result = VerificationResult.INVALID
        
        return VerificationReport(
            result=result,
            details=details,
            errors=errors,
            warnings=warnings,
            reasoning_steps=[],
            execution_trace=[]
        )
    
    def _verify_goal_satisfaction(self, state: Set[str], goal_state: Set[str]) -> VerificationReport:
        """Verify if goal is satisfied"""
        details = []
        errors = []
        warnings = []
        
        unsatisfied_goals = goal_state - state
        
        if not unsatisfied_goals:
            details.append("All goals are satisfied")
            result = VerificationResult.VALID
        else:
            errors.append(f"Unsatisfied goals: {', '.join(unsatisfied_goals)}")
            result = VerificationResult.INVALID
        
        return VerificationReport(
            result=result,
            details=details,
            errors=errors,
            warnings=warnings,
            reasoning_steps=[],
            execution_trace=[]
        )
    
    def _instantiate_literal(self, literal: str, parameters: Dict[str, str]) -> str:
        """Instantiate a literal with parameter values"""
        instantiated = literal
        for param_name, param_value in parameters.items():
            instantiated = instantiated.replace(f"?{param_name}", param_value)
        return instantiated
    
    def _is_literal_satisfied(self, literal: str, state: Set[str]) -> bool:
        """Check if a literal is satisfied in a state"""
        if literal.startswith("(not "):
            # Negative literal
            positive_literal = literal[5:-1]  # Remove "(not " and ")"
            return positive_literal not in state
        else:
            # Positive literal
            return literal in state
    
    def _parse_action_string(self, action_str: str) -> Tuple[str, Dict[str, str]]:
        """Parse an action string into action name and parameters"""
        match = re.match(r'(\w+)\(([^)]+)\)', action_str)
        if match:
            action_name = match.group(1)
            param_str = match.group(2)
            parameters = {}
            
            if param_str.strip():
                param_values = [p.strip() for p in param_str.split(',')]
                # Map parameter values to parameter names
                if action_name in self.domain.actions:
                    action = self.domain.actions[action_name]
                    for i, param_value in enumerate(param_values):
                        if i < len(action.parameters):
                            param_name = action.parameters[i][0]
                            parameters[param_name] = param_value
            
            return action_name, parameters
        
        raise ValueError(f"Could not parse action string: {action_str}")


class InvariantChecker:
    """Checks for invariant preservation during state transitions"""
    
    def __init__(self, domain: PDDLDomain):
        self.domain = domain
        self.invariants = []
        self._extract_invariants()
    
    def _extract_invariants(self):
        """Extract invariants from domain predicates"""
        # This is a simplified invariant extraction
        # In practice, you'd need more sophisticated invariant detection
        
        # Example: if we have a predicate like (on ?x ?y), we might want to check
        # that objects don't float in mid-air
        for pred_name, predicate in self.domain.predicates.items():
            if pred_name == "on" and len(predicate.parameters) == 2:
                # Add invariant: if something is on something else, the bottom object should be clear
                self.invariants.append({
                    "name": f"on_invariant_{pred_name}",
                    "condition": f"(on ?x ?y)",
                    "implication": f"(clear ?y)",
                    "description": "If x is on y, then y should be clear"
                })
    
    def check_invariants(self, state: Set[str]) -> List[Dict[str, Any]]:
        """Check if invariants are preserved in a state"""
        violations = []
        
        for invariant in self.invariants:
            # This is a simplified invariant checking
            # In practice, you'd need more sophisticated logic
            if not self._check_invariant(invariant, state):
                violations.append({
                    "invariant": invariant,
                    "violation": f"Invariant {invariant['name']} is violated",
                    "state": state
                })
        
        return violations
    
    def _check_invariant(self, invariant: Dict[str, Any], state: Set[str]) -> bool:
        """Check a specific invariant"""
        # Simplified invariant checking
        # In practice, you'd need more sophisticated logic
        return True  # Placeholder


# Example usage
if __name__ == "__main__":
    from pddl_parser import PDDLParser
    
    # Create a simple domain for testing
    parser = PDDLParser()
    domain_text = """
    (define (domain blocksworld)
      (:requirements :strips)
      (:predicates
        (on ?x ?y)
        (clear ?x)
        (ontable ?x)
        (handempty)
        (holding ?x)
      )
      (:action pickup
        :parameters (?x)
        :precondition (and (clear ?x) (ontable ?x) (handempty))
        :effect (and (holding ?x) (not (ontable ?x)) (not (clear ?x)) (not (handempty)))
      )
    )
    """
    
    domain = parser.parse_domain(domain_text)
    verifier = StateVerifier(domain)
    
    # Test action applicability verification
    state = {"(clear a)", "(ontable a)", "(handempty)"}
    report = verifier.verify_action_applicability("pickup", {"x": "a"}, state)
    
    print("Action Applicability Verification:")
    print(f"Result: {report.result.value}")
    print(f"Details: {report.details}")
    if report.errors:
        print(f"Errors: {report.errors}")
    if report.warnings:
        print(f"Warnings: {report.warnings}")
