"""
PDDL Parser for Symbolic Planning
Implements parsing of PDDL domain and problem files for the PDDL-INSTRUCT framework.
"""

import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class PDDLType(Enum):
    """PDDL element types"""
    DOMAIN = "domain"
    PROBLEM = "problem"
    ACTION = "action"
    PREDICATE = "predicate"
    OBJECT = "object"
    INIT = "init"
    GOAL = "goal"


@dataclass
class PDDLPredicate:
    """Represents a PDDL predicate"""
    name: str
    parameters: List[Tuple[str, str]]  # (param_name, param_type)
    
    def __str__(self):
        params = " ".join([f"{name} - {type_name}" for name, type_name in self.parameters])
        return f"({self.name} {params})"


@dataclass
class PDDLAction:
    """Represents a PDDL action"""
    name: str
    parameters: List[Tuple[str, str]]  # (param_name, param_type)
    preconditions: List[str]  # List of precondition literals
    effects: List[str]  # List of effect literals
    
    def __str__(self):
        params = " ".join([f"{name} - {type_name}" for name, type_name in self.parameters])
        preconds = " ".join(self.preconditions)
        effects = " ".join(self.effects)
        return f"(:action {self.name}\n  :parameters ({params})\n  :precondition (and {preconds})\n  :effect (and {effects}))"


@dataclass
class PDDLDomain:
    """Represents a PDDL domain"""
    name: str
    requirements: List[str]
    types: Dict[str, str]  # type_name -> parent_type
    predicates: Dict[str, PDDLPredicate]
    actions: Dict[str, PDDLAction]
    
    def __str__(self):
        return f"Domain: {self.name}"


@dataclass
class PDDLProblem:
    """Represents a PDDL problem"""
    name: str
    domain_name: str
    objects: Dict[str, str]  # object_name -> type_name
    init_state: Set[str]  # Set of initial state literals
    goal_state: Set[str]  # Set of goal state literals
    
    def __str__(self):
        return f"Problem: {self.name} (Domain: {self.domain_name})"


class PDDLParser:
    """Parser for PDDL domain and problem files"""
    
    def __init__(self):
        self.current_domain = None
        self.current_problem = None
    
    def parse_domain(self, domain_text: str) -> PDDLDomain:
        """Parse a PDDL domain file"""
        # Remove comments and normalize whitespace
        domain_text = self._clean_text(domain_text)
        
        # Extract domain name
        domain_name = self._extract_domain_name(domain_text)
        
        # Extract requirements
        requirements = self._extract_requirements(domain_text)
        
        # Extract types
        types = self._extract_types(domain_text)
        
        # Extract predicates
        predicates = self._extract_predicates(domain_text)
        
        # Extract actions
        actions = self._extract_actions(domain_text)
        
        domain = PDDLDomain(
            name=domain_name,
            requirements=requirements,
            types=types,
            predicates=predicates,
            actions=actions
        )
        
        self.current_domain = domain
        return domain
    
    def parse_problem(self, problem_text: str) -> PDDLProblem:
        """Parse a PDDL problem file"""
        # Remove comments and normalize whitespace
        problem_text = self._clean_text(problem_text)
        
        # Extract problem name
        problem_name = self._extract_problem_name(problem_text)
        
        # Extract domain name
        domain_name = self._extract_problem_domain(problem_text)
        
        # Extract objects
        objects = self._extract_objects(problem_text)
        
        # Extract initial state
        init_state = self._extract_init_state(problem_text)
        
        # Extract goal state
        goal_state = self._extract_goal_state(problem_text)
        
        problem = PDDLProblem(
            name=problem_name,
            domain_name=domain_name,
            objects=objects,
            init_state=init_state,
            goal_state=goal_state
        )
        
        self.current_problem = problem
        return problem
    
    def _clean_text(self, text: str) -> str:
        """Remove comments and normalize whitespace"""
        # Remove comments
        text = re.sub(r';.*$', '', text, flags=re.MULTILINE)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_domain_name(self, text: str) -> str:
        """Extract domain name from domain file"""
        # Look for (define (domain <name>) pattern
        match = re.search(r'\(define\s+\(domain\s+(\w+)\)', text)
        if match:
            return match.group(1)
        raise ValueError("Could not find domain name")
    
    def _extract_requirements(self, text: str) -> List[str]:
        """Extract requirements from domain file"""
        match = re.search(r'\(:requirements\s+([^)]+)\)', text)
        if match:
            return match.group(1).split()
        return []
    
    def _extract_types(self, text: str) -> Dict[str, str]:
        """Extract types from domain file"""
        types = {}
        match = re.search(r'\(:types\s+([^)]+)\)', text)
        if match:
            type_defs = match.group(1).split()
            for i in range(0, len(type_defs), 2):
                if i + 1 < len(type_defs):
                    types[type_defs[i]] = type_defs[i + 1]
        return types
    
    def _extract_predicates(self, text: str) -> Dict[str, PDDLPredicate]:
        """Extract predicates from domain file"""
        predicates = {}
        # Find all predicate definitions
        pred_matches = re.finditer(r'\(:predicates\s+([^)]+)\)', text)
        for match in pred_matches:
            pred_text = match.group(1)
            # Parse individual predicates
            pred_defs = re.findall(r'\((\w+)([^)]*)\)', pred_text)
            for name, params in pred_defs:
                param_list = []
                if params.strip():
                    param_parts = params.strip().split()
                    for i in range(0, len(param_parts), 2):
                        if i + 1 < len(param_parts):
                            param_list.append((param_parts[i], param_parts[i + 1]))
                predicates[name] = PDDLPredicate(name, param_list)
        return predicates
    
    def _extract_actions(self, text: str) -> Dict[str, PDDLAction]:
        """Extract actions from domain file"""
        actions = {}
        # Find all action definitions
        action_matches = re.finditer(r'\(:action\s+(\w+)\s+([^)]+)\)', text, re.DOTALL)
        for match in action_matches:
            name = match.group(1)
            action_text = match.group(2)
            
            # Extract parameters
            params = self._extract_action_parameters(action_text)
            
            # Extract preconditions
            preconditions = self._extract_preconditions(action_text)
            
            # Extract effects
            effects = self._extract_effects(action_text)
            
            actions[name] = PDDLAction(name, params, preconditions, effects)
        
        return actions
    
    def _extract_action_parameters(self, action_text: str) -> List[Tuple[str, str]]:
        """Extract parameters from action definition"""
        params = []
        match = re.search(r':parameters\s+\(([^)]+)\)', action_text)
        if match:
            param_text = match.group(1)
            param_parts = param_text.split()
            for i in range(0, len(param_parts), 2):
                if i + 1 < len(param_parts):
                    params.append((param_parts[i], param_parts[i + 1]))
        return params
    
    def _extract_preconditions(self, action_text: str) -> List[str]:
        """Extract preconditions from action definition"""
        preconditions = []
        match = re.search(r':precondition\s+\(and\s+([^)]+)\)', action_text)
        if match:
            prec_text = match.group(1)
            # Split by parentheses to get individual preconditions
            prec_matches = re.findall(r'\([^)]+\)', prec_text)
            preconditions.extend(prec_matches)
        return preconditions
    
    def _extract_effects(self, action_text: str) -> List[str]:
        """Extract effects from action definition"""
        effects = []
        match = re.search(r':effect\s+\(and\s+([^)]+)\)', action_text)
        if match:
            effect_text = match.group(1)
            # Split by parentheses to get individual effects
            effect_matches = re.findall(r'\([^)]+\)', effect_text)
            effects.extend(effect_matches)
        return effects
    
    def _extract_problem_name(self, text: str) -> str:
        """Extract problem name from problem file"""
        # Look for (define (problem <name>) pattern
        match = re.search(r'\(define\s+\(problem\s+([\w-]+)\)', text)
        if match:
            return match.group(1)
        raise ValueError("Could not find problem name")
    
    def _extract_problem_domain(self, text: str) -> str:
        """Extract domain name from problem file"""
        match = re.search(r'\(:domain\s+(\w+)\)', text)
        if match:
            return match.group(1)
        raise ValueError("Could not find domain name in problem")
    
    def _extract_objects(self, text: str) -> Dict[str, str]:
        """Extract objects from problem file"""
        objects = {}
        match = re.search(r'\(:objects\s+([^)]+)\)', text)
        if match:
            obj_text = match.group(1)
            obj_parts = obj_text.split()
            for i in range(0, len(obj_parts), 2):
                if i + 1 < len(obj_parts):
                    objects[obj_parts[i]] = obj_parts[i + 1]
        return objects
    
    def _extract_init_state(self, text: str) -> Set[str]:
        """Extract initial state from problem file"""
        init_state = set()
        # Look for :init section with multiline support
        match = re.search(r'\(:init\s+([\s\S]*?)\)\s*\(:goal', text)
        if match:
            init_text = match.group(1)
            # Find all literals in initial state
            literals = re.findall(r'\([^)]+\)', init_text)
            init_state.update(literals)
        return init_state
    
    def _extract_goal_state(self, text: str) -> Set[str]:
        """Extract goal state from problem file"""
        goal_state = set()
        # Find the start of :goal
        goal_start = text.find('(:goal')
        if goal_start != -1:
            # Find the matching closing parenthesis
            paren_count = 0
            goal_end = goal_start
            for i, char in enumerate(text[goal_start:], goal_start):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        goal_end = i + 1
                        break
            
            goal_section = text[goal_start:goal_end]
            # Find all literals directly in goal section, but exclude the goal and and parts
            literals = re.findall(r'\([^)]+\)', goal_section)
            # Filter out the goal and and parts
            filtered_literals = [lit for lit in literals if not lit.startswith('(:goal') and not lit.startswith('(and')]
            goal_state.update(filtered_literals)
        return goal_state


def load_pddl_file(file_path: str) -> str:
    """Load PDDL file content"""
    with open(file_path, 'r') as f:
        return f.read()


# Example usage
if __name__ == "__main__":
    parser = PDDLParser()
    
    # Example domain
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
    print(f"Parsed domain: {domain.name}")
    print(f"Predicates: {list(domain.predicates.keys())}")
    print(f"Actions: {list(domain.actions.keys())}")
