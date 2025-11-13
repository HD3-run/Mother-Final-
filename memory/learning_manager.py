"""
Learning Manager for MOTHER

This module implements recursive sub-goaling for learning. When MOTHER
encounters a concept it doesn't understand, it breaks it down into
prerequisites and learns those first.
"""

import json
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from memory.knowledge_graph import KnowledgeGraph, ConceptNode

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Status of a learning goal"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class LearningGoal:
    """Represents a single learning goal"""
    concept: str
    status: GoalStatus = GoalStatus.PENDING
    prerequisites: List[str] = field(default_factory=list)
    sub_goals: List['LearningGoal'] = field(default_factory=list)
    parent_goal: Optional['LearningGoal'] = None
    depth: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "concept": self.concept,
            "status": self.status.value,
            "prerequisites": self.prerequisites,
            "sub_goals": [sg.to_dict() for sg in self.sub_goals],
            "depth": self.depth,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LearningGoal':
        """Deserialize from dictionary"""
        goal = cls(
            concept=data["concept"],
            status=GoalStatus(data["status"]),
            prerequisites=data.get("prerequisites", []),
            depth=data.get("depth", 0),
        )
        goal.sub_goals = [cls.from_dict(sg) for sg in data.get("sub_goals", [])]
        return goal


class LearningGoalManager:
    """Manages learning goals with recursive sub-goaling"""
    
    def __init__(self, graph: KnowledgeGraph, config: dict | None = None):
        """Initialize the learning goal manager.
        
        Args:
            graph: The KnowledgeGraph instance to check for existing knowledge.
            config: Configuration dictionary (optional).
        """
        self.graph = graph
        self.config = config or {}
        self.goals: List[LearningGoal] = []
        self.max_depth = self.config.get("max_learning_depth", 5)
    
    def add_learning_goal(self, concept: str, parent: Optional[LearningGoal] = None) -> LearningGoal:
        """Add a new learning goal for a concept.
        
        Args:
            concept: The concept to learn.
            parent: The parent goal if this is a sub-goal.
            
        Returns:
            The created LearningGoal instance.
        """
        # Check if goal already exists
        existing = self._find_goal(concept)
        if existing:
            logger.info(f"[Learning Manager]: Goal already exists for '{concept}'")
            return existing
        
        depth = (parent.depth + 1) if parent else 0
        if depth > self.max_depth:
            logger.warning(f"[Learning Manager]: Max depth reached for '{concept}'")
            goal = LearningGoal(concept, GoalStatus.BLOCKED, depth=depth)
            if parent:
                parent.sub_goals.append(goal)
                goal.parent_goal = parent
            return goal
        
        goal = LearningGoal(concept, depth=depth)
        if parent:
            parent.sub_goals.append(goal)
            goal.parent_goal = parent
        else:
            self.goals.append(goal)
        
        logger.info(f"[Learning Manager]: Added learning goal for '{concept}' (depth: {depth})")
        return goal
    
    def _find_goal(self, concept: str) -> Optional[LearningGoal]:
        """Find a goal by concept name"""
        for goal in self.goals:
            found = self._find_goal_recursive(goal, concept)
            if found:
                return found
        return None
    
    def _find_goal_recursive(self, goal: LearningGoal, concept: str) -> Optional[LearningGoal]:
        """Recursively search for a goal"""
        if goal.concept.lower() == concept.lower():
            return goal
        for sub_goal in goal.sub_goals:
            found = self._find_goal_recursive(sub_goal, concept)
            if found:
                return found
        return None
    
    def identify_prerequisites(
        self,
        concept: str,
        config: dict | None = None,
    ) -> List[str]:
        """
        Uses LLM to identify prerequisites for learning a concept.
        
        Args:
            concept: The concept to learn.
            config: Configuration dictionary for LLM (optional).
            
        Returns:
            A list of prerequisite concept names.
        """
        logger.info(f"[Learning Manager]: Identifying prerequisites for '{concept}'")
        
        if config is None:
            config = self.config
        
        try:
            from processing.llm_handler import get_response
            
            system_prompt = (
                "You are a learning prerequisite analyzer. Given a concept to learn, "
                "identify the fundamental concepts that must be understood first. "
                "Output ONLY a JSON array of prerequisite concept names, or an empty array [] if none are needed."
            )
            
            prompt = (
                f"{system_prompt}\n\n"
                f"Concept to learn: {concept}\n"
                f"Output (JSON array):"
            )
            
            llm_config = config.copy()
            llm_config["temperature"] = 0.3
            llm_config["max_tokens"] = 100
            
            response = get_response(
                prompt=prompt,
                config=llm_config,
                system_context=system_prompt,
            )
            
            if not response:
                return []
            
            # Extract JSON array
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                prerequisites = json.loads(json_match.group())
                logger.info(f"    - Found {len(prerequisites)} prerequisites: {prerequisites}")
                return prerequisites
            else:
                return []
                
        except Exception as e:
            logger.error(f"[Learning Manager Error]: Failed to identify prerequisites: {e}")
            return []
    
    def check_concept_understood(self, concept: str) -> bool:
        """Check if a concept is already understood (exists in graph with sufficient connections).
        
        Args:
            concept: The concept to check.
            
        Returns:
            True if the concept is understood, False otherwise.
        """
        node = self.graph.get_node_by_name(concept)
        if not node:
            return False
        
        # Check if node has meaningful connections
        outgoing = self.graph.get_edges_from_node(node.id)
        incoming = self.graph.get_edges_to_node(node.id)
        
        # Consider a concept "understood" if it has at least 2 connections
        return len(outgoing) + len(incoming) >= 2
    
    def build_learning_plan(self, concept: str) -> LearningGoal:
        """Build a recursive learning plan for a concept.
        
        Args:
            concept: The concept to learn.
            
        Returns:
            The root LearningGoal with all sub-goals.
        """
        logger.info(f"[Learning Manager]: Building learning plan for '{concept}'")
        
        root_goal = self.add_learning_goal(concept)
        root_goal.status = GoalStatus.IN_PROGRESS
        
        # Recursively build sub-goals
        self._build_learning_plan_recursive(root_goal)
        
        return root_goal
    
    def _build_learning_plan_recursive(self, goal: LearningGoal) -> None:
        """Recursively build learning plan"""
        # Check if already understood
        if self.check_concept_understood(goal.concept):
            goal.status = GoalStatus.COMPLETED
            logger.info(f"    - Concept '{goal.concept}' already understood")
            return
        
        # Identify prerequisites
        prerequisites = self.identify_prerequisites(goal.concept)
        goal.prerequisites = prerequisites
        
        # Create sub-goals for each prerequisite
        for prereq in prerequisites:
            if not self.check_concept_understood(prereq):
                sub_goal = self.add_learning_goal(prereq, parent=goal)
                sub_goal.status = GoalStatus.PENDING
                # Recursively build plan for sub-goal
                self._build_learning_plan_recursive(sub_goal)
            else:
                logger.info(f"    - Prerequisite '{prereq}' already understood")
        
        # Mark as in progress if we have a plan
        if goal.sub_goals:
            goal.status = GoalStatus.IN_PROGRESS
        else:
            # No prerequisites, can learn directly
            goal.status = GoalStatus.PENDING
    
    def get_next_goal_to_learn(self) -> Optional[LearningGoal]:
        """Get the next goal that should be learned (lowest depth, pending status).
        
        Returns:
            The next LearningGoal to work on, or None if all goals are completed.
        """
        def find_next_recursive(goal: LearningGoal) -> Optional[LearningGoal]:
            # If this goal is pending and has no incomplete sub-goals, return it
            if goal.status == GoalStatus.PENDING:
                incomplete_subgoals = [sg for sg in goal.sub_goals if sg.status != GoalStatus.COMPLETED]
                if not incomplete_subgoals:
                    return goal
            
            # Otherwise, check sub-goals
            for sub_goal in goal.sub_goals:
                if sub_goal.status != GoalStatus.COMPLETED:
                    found = find_next_recursive(sub_goal)
                    if found:
                        return found
            
            return None
        
        for goal in self.goals:
            if goal.status != GoalStatus.COMPLETED:
                found = find_next_recursive(goal)
                if found:
                    return found
        
        return None
    
    def mark_goal_completed(self, concept: str) -> None:
        """Mark a learning goal as completed.
        
        Args:
            concept: The concept that was learned.
        """
        goal = self._find_goal(concept)
        if goal:
            goal.status = GoalStatus.COMPLETED
            logger.info(f"[Learning Manager]: Marked '{concept}' as completed")
            
            # Check if parent goal can now proceed
            if goal.parent_goal:
                all_prereqs_done = all(
                    sg.status == GoalStatus.COMPLETED
                    for sg in goal.parent_goal.sub_goals
                )
                if all_prereqs_done and goal.parent_goal.status == GoalStatus.BLOCKED:
                    goal.parent_goal.status = GoalStatus.PENDING
    
    def save_goals(self, filename: str) -> None:
        """Save learning goals to a JSON file.
        
        Args:
            filename: Path to the file to save to.
        """
        data = {
            "goals": [goal.to_dict() for goal in self.goals],
            "max_depth": self.max_depth,
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"[Learning Manager]: Saved goals to {filename}")
    
    def load_goals(self, filename: str) -> None:
        """Load learning goals from a JSON file.
        
        Args:
            filename: Path to the file to load from.
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.goals = [LearningGoal.from_dict(g) for g in data.get("goals", [])]
            self.max_depth = data.get("max_depth", 5)
            logger.info(f"[Learning Manager]: Loaded {len(self.goals)} goals from {filename}")
        except FileNotFoundError:
            logger.info(f"[Learning Manager]: No saved goals found at {filename}")
        except Exception as e:
            logger.error(f"[Learning Manager Error]: Failed to load goals: {e}")

