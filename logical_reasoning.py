"""
Logical Chain-of-Thought Reasoning Framework
Implements the core reasoning components for PDDL-INSTRUCT framework.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re
from pddl_parser import PDDLDomain, PDDLProblem, PDDLAction


class ReasoningStepType(Enum):
    """Types of reasoning steps in the chain-of-thought"""
    PRECONDITION_CHECK = "precondition_check"
    EFFECT_APPLICATION = "effect_application"
    STATE_TRANSITION = "state_transition"
    GOAL_VERIFICATION = "goal_verification"
    ACTION_APPLICABILITY = "action_applicability"
    INVARIANT_PRESERVATION = "invariant_preservation"


@dataclass
class ReasoningStep:
    """Represents a single reasoning step"""
    step_type: ReasoningStepType
    description: str
    input_state: Set[str]
    output_state: Set[str]
    action: Optional[str] = None
    parameters: Optional[Dict[str, str]] = None
    success: bool = True
    explanation: str = ""


@dataclass
class ReasoningChain:
    """Represents a complete chain of reasoning steps"""
    steps: List[ReasoningStep]
    final_state: Set[str]
    plan: List[str]
    success: bool
    explanation: str


class StateManager:
    """Manages state transitions and reasoning about states"""
    
    def __init__(self, domain: PDDLDomain):
        self.domain = domain
        self.current_state = set()
        self.reasoning_steps = []
    
    def set_state(self, state: Set[str]):
        """Set the current state"""
        self.current_state = state.copy()
    
    def get_state(self) -> Set[str]:
        """Get the current state"""
        return self.current_state.copy()
    
    def apply_action(self, action_name: str, parameters: Dict[str, str]) -> Tuple[bool, str, Set[str]]:
        """Apply an action to the current state"""
        if action_name not in self.domain.actions:
            return False, f"Action {action_name} not found in domain", self.current_state
        
        action = self.domain.actions[action_name]
        
        # Check preconditions
        precond_success, precond_explanation = self._check_preconditions(action, parameters)
        if not precond_success:
            return False, f"Preconditions not met: {precond_explanation}", self.current_state
        
        # Apply effects
        new_state = self._apply_effects(action, parameters)
        
        # Record reasoning step
        step = ReasoningStep(
            step_type=ReasoningStepType.ACTION_APPLICABILITY,
            description=f"Apply action {action_name} with parameters {parameters}",
            input_state=self.current_state.copy(),
            output_state=new_state.copy(),
            action=action_name,
            parameters=parameters,
            success=True,
            explanation=f"Successfully applied {action_name}: {precond_explanation}"
        )
        self.reasoning_steps.append(step)
        
        self.current_state = new_state
        return True, f"Successfully applied {action_name}", new_state
    
    def _check_preconditions(self, action: PDDLAction, parameters: Dict[str, str]) -> Tuple[bool, str]:
        """Check if action preconditions are satisfied"""
        explanations = []
        
        for precondition in action.preconditions:
            # Instantiate precondition with parameters
            instantiated_precond = self._instantiate_literal(precondition, parameters)
            
            # Check if precondition is satisfied
            if not self._is_literal_satisfied(instantiated_precond):
                explanations.append(f"Precondition {instantiated_precond} not satisfied")
                return False, "; ".join(explanations)
            else:
                explanations.append(f"Precondition {instantiated_precond} satisfied")
        
        return True, "; ".join(explanations)
    
    def _apply_effects(self, action: PDDLAction, parameters: Dict[str, str]) -> Set[str]:
        """Apply action effects to current state"""
        new_state = self.current_state.copy()
        
        for effect in action.effects:
            # Instantiate effect with parameters
            instantiated_effect = self._instantiate_literal(effect, parameters)
            
            # Apply positive or negative effect
            if instantiated_effect.startswith("(not "):
                # Negative effect - remove literal
                positive_literal = instantiated_effect[5:-1]  # Remove "(not " and ")"
                new_state.discard(positive_literal)
            else:
                # Positive effect - add literal
                new_state.add(instantiated_effect)
        
        return new_state
    
    def _instantiate_literal(self, literal: str, parameters: Dict[str, str]) -> str:
        """Instantiate a literal with parameter values"""
        instantiated = literal
        for param_name, param_value in parameters.items():
            instantiated = instantiated.replace(f"?{param_name}", param_value)
        return instantiated
    
    def _is_literal_satisfied(self, literal: str) -> bool:
        """Check if a literal is satisfied in current state"""
        if literal.startswith("(not "):
            # Negative literal
            positive_literal = literal[5:-1]  # Remove "(not " and ")"
            return positive_literal not in self.current_state
        else:
            # Positive literal
            return literal in self.current_state


class LogicalReasoner:
    """Main logical reasoning engine for PDDL-INSTRUCT"""
    
    def __init__(self, domain: PDDLDomain):
        self.domain = domain
        self.state_manager = StateManager(domain)
        self.reasoning_chains = []
    
    def reason_about_action_applicability(self, action_name: str, parameters: Dict[str, str], 
                                        current_state: Set[str]) -> ReasoningChain:
        """Reason about whether an action is applicable in a given state"""
        self.state_manager.set_state(current_state)
        
        steps = []
        explanations = []
        
        if action_name not in self.domain.actions:
            return ReasoningChain(
                steps=[],
                final_state=current_state,
                plan=[],
                success=False,
                explanation=f"Action {action_name} not found in domain"
            )
        
        action = self.domain.actions[action_name]
        
        # Step 1: Check preconditions
        precond_step = self._create_precondition_step(action, parameters, current_state)
        steps.append(precond_step)
        explanations.append(precond_step.explanation)
        
        if not precond_step.success:
            return ReasoningChain(
                steps=steps,
                final_state=current_state,
                plan=[],
                success=False,
                explanation="; ".join(explanations)
            )
        
        # Step 2: Apply effects
        effect_step = self._create_effect_step(action, parameters, current_state)
        steps.append(effect_step)
        explanations.append(effect_step.explanation)
        
        return ReasoningChain(
            steps=steps,
            final_state=effect_step.output_state,
            plan=[f"{action_name}({', '.join(parameters.values())})"],
            success=True,
            explanation="; ".join(explanations)
        )
    
    def reason_about_plan_validity(self, plan: List[str], initial_state: Set[str], 
                                 goal_state: Set[str]) -> ReasoningChain:
        """Reason about the validity of a complete plan"""
        self.state_manager.set_state(initial_state)
        steps = []
        explanations = []
        current_state = initial_state.copy()
        
        for action_str in plan:
            # Parse action string
            action_name, parameters = self._parse_action_string(action_str)
            
            # Reason about action applicability
            action_chain = self.reason_about_action_applicability(action_name, parameters, current_state)
            steps.extend(action_chain.steps)
            explanations.append(action_chain.explanation)
            
            if not action_chain.success:
                return ReasoningChain(
                    steps=steps,
                    final_state=current_state,
                    plan=plan,
                    success=False,
                    explanation="; ".join(explanations)
                )
            
            current_state = action_chain.final_state
        
        # Step 3: Check goal satisfaction
        goal_step = self._create_goal_verification_step(current_state, goal_state)
        steps.append(goal_step)
        explanations.append(goal_step.explanation)
        
        return ReasoningChain(
            steps=steps,
            final_state=current_state,
            plan=plan,
            success=goal_step.success,
            explanation="; ".join(explanations)
        )
    
    def _create_precondition_step(self, action: PDDLAction, parameters: Dict[str, str], 
                                state: Set[str]) -> ReasoningStep:
        """Create a reasoning step for precondition checking"""
        precond_success, precond_explanation = self.state_manager._check_preconditions(action, parameters)
        
        return ReasoningStep(
            step_type=ReasoningStepType.PRECONDITION_CHECK,
            description=f"Check preconditions for {action.name}",
            input_state=state.copy(),
            output_state=state.copy(),
            action=action.name,
            parameters=parameters,
            success=precond_success,
            explanation=precond_explanation
        )
    
    def _create_effect_step(self, action: PDDLAction, parameters: Dict[str, str], 
                          state: Set[str]) -> ReasoningStep:
        """Create a reasoning step for effect application"""
        new_state = self.state_manager._apply_effects(action, parameters)
        
        return ReasoningStep(
            step_type=ReasoningStepType.EFFECT_APPLICATION,
            description=f"Apply effects of {action.name}",
            input_state=state.copy(),
            output_state=new_state,
            action=action.name,
            parameters=parameters,
            success=True,
            explanation=f"Applied effects: {', '.join(action.effects)}"
        )
    
    def _create_goal_verification_step(self, state: Set[str], goal_state: Set[str]) -> ReasoningStep:
        """Create a reasoning step for goal verification"""
        unsatisfied_goals = goal_state - state
        success = len(unsatisfied_goals) == 0
        
        if success:
            explanation = "All goals satisfied"
        else:
            explanation = f"Unsatisfied goals: {', '.join(unsatisfied_goals)}"
        
        return ReasoningStep(
            step_type=ReasoningStepType.GOAL_VERIFICATION,
            description="Verify goal satisfaction",
            input_state=state.copy(),
            output_state=state.copy(),
            success=success,
            explanation=explanation
        )
    
    def _parse_action_string(self, action_str: str) -> Tuple[str, Dict[str, str]]:
        """Parse an action string into action name and parameters"""
        # Simple parsing - assumes format "action_name(param1, param2, ...)"
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
    
    def generate_reasoning_chain(self, problem: PDDLProblem, plan: List[str]) -> ReasoningChain:
        """Generate a complete reasoning chain for a problem and plan"""
        return self.reason_about_plan_validity(plan, problem.init_state, problem.goal_state)


class InstructionGenerator:
    """Generates instruction templates for training LLMs"""
    
    def __init__(self, domain: PDDLDomain):
        self.domain = domain
        self.reasoner = LogicalReasoner(domain)
    
    def generate_action_applicability_instruction(self, action_name: str, parameters: Dict[str, str], 
                                                state: Set[str]) -> str:
        """Generate instruction for reasoning about action applicability"""
        reasoning_chain = self.reasoner.reason_about_action_applicability(action_name, parameters, state)
        
        instruction = f"""Given the following PDDL domain and current state, determine if the action {action_name} is applicable.

Domain: {self.domain.name}
Current State: {', '.join(sorted(state))}

Action: {action_name}
Parameters: {parameters}

Please reason step by step:
1. Check each precondition of the action
2. Verify if all preconditions are satisfied in the current state
3. If applicable, describe what effects the action would have

Reasoning Chain:
"""
        
        for i, step in enumerate(reasoning_chain.steps, 1):
            instruction += f"{i}. {step.description}: {step.explanation}\n"
        
        instruction += f"\nConclusion: Action {action_name} is {'applicable' if reasoning_chain.success else 'not applicable'}."
        
        return instruction
    
    def generate_plan_validation_instruction(self, problem: PDDLProblem, plan: List[str]) -> str:
        """Generate instruction for plan validation"""
        reasoning_chain = self.reasoner.generate_reasoning_chain(problem, plan)
        
        instruction = f"""Given the following PDDL problem and proposed plan, validate whether the plan achieves the goal.

Problem: {problem.name}
Domain: {problem.domain_name}
Initial State: {', '.join(sorted(problem.init_state))}
Goal State: {', '.join(sorted(problem.goal_state))}

Proposed Plan: {', '.join(plan)}

Please reason step by step:
1. Start with the initial state
2. For each action in the plan:
   a. Check if the action is applicable in the current state
   b. Apply the action's effects to get the new state
3. Verify if the final state satisfies all goals

Reasoning Chain:
"""
        
        for i, step in enumerate(reasoning_chain.steps, 1):
            instruction += f"{i}. {step.description}: {step.explanation}\n"
        
        instruction += f"\nConclusion: The plan is {'valid' if reasoning_chain.success else 'invalid'}."
        
        return instruction


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
    reasoner = LogicalReasoner(domain)
    
    # Test action applicability reasoning
    state = {"(clear a)", "(ontable a)", "(handempty)"}
    reasoning_chain = reasoner.reason_about_action_applicability("pickup", {"x": "a"}, state)
    
    print("Action Applicability Reasoning:")
    print(f"Success: {reasoning_chain.success}")
    print(f"Explanation: {reasoning_chain.explanation}")
    print(f"Final State: {reasoning_chain.final_state}")
