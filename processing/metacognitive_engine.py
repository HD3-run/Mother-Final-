"""
Metacognitive Engine for MOTHER

This module provides self-analysis and self-improvement capabilities,
allowing MOTHER to reflect on its own cognitive processes, identify
patterns, and optimize its behavior over time.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from processing.llm_handler import get_response
from memory.knowledge_graph import KnowledgeGraph
from memory.structured_store import get_knowledge_graph
from personality.loader import load_config

logger = logging.getLogger(__name__)

METACOGNITIVE_STATE_FILE = Path("data/metacognitive_state.json")


class MetacognitiveEngine:
    """Enables MOTHER to think about its own thinking.
    
    This engine provides:
    - Performance analysis (accuracy, response quality)
    - Pattern recognition (common mistakes, successful strategies)
    - Self-optimization (adjusting parameters based on performance)
    - Reflection on cognitive processes
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Metacognitive Engine.
        
        Args:
            config: Configuration dictionary (optional).
        """
        self.config = config or load_config()
        self.graph = get_knowledge_graph()
        self.state = self._load_state()
        logger.info("[Metacognitive Engine] Initialized")
    
    def _load_state(self) -> Dict[str, Any]:
        """Load metacognitive state from disk."""
        if METACOGNITIVE_STATE_FILE.exists():
            try:
                with open(METACOGNITIVE_STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[Metacognitive] Failed to load state: {e}")
        
        return {
            "performance_metrics": {
                "total_interactions": 0,
                "successful_interpretations": 0,
                "failed_interpretations": 0,
                "clarification_requests": 0,
                "knowledge_additions": 0,
            },
            "patterns": {
                "common_errors": [],
                "successful_strategies": [],
                "user_preferences": {},
            },
            "optimizations": {
                "confidence_threshold": 0.7,
                "interpretation_temperature": 0.0,
                "synthesis_temperature": 0.1,
            },
            "last_analysis": None,
        }
    
    def _save_state(self):
        """Save metacognitive state to disk."""
        try:
            with open(METACOGNITIVE_STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"[Metacognitive] Failed to save state: {e}")
    
    def record_interaction(
        self,
        success: bool,
        interaction_type: str = "interpretation",
        metadata: Optional[Dict] = None,
    ):
        """Record an interaction for later analysis.
        
        Args:
            success: Whether the interaction was successful.
            interaction_type: Type of interaction (interpretation, learning, etc.)
            metadata: Additional context about the interaction.
        """
        metrics = self.state["performance_metrics"]
        metrics["total_interactions"] += 1
        
        if success:
            metrics["successful_interpretations"] += 1
        else:
            metrics["failed_interpretations"] += 1
        
        if metadata:
            if "needed_clarification" in metadata:
                metrics["clarification_requests"] += 1
            if "knowledge_added" in metadata:
                metrics["knowledge_additions"] += 1
        
        self._save_state()
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall performance and identify patterns.
        
        Returns:
            A dictionary containing performance analysis and recommendations.
        """
        logger.info("[Metacognitive] Analyzing performance...")
        
        metrics = self.state["performance_metrics"]
        total = metrics["total_interactions"]
        
        if total == 0:
            return {
                "status": "insufficient_data",
                "message": "Not enough interactions to analyze yet.",
            }
        
        success_rate = metrics["successful_interpretations"] / total
        failure_rate = metrics["failed_interpretations"] / total
        clarification_rate = metrics["clarification_requests"] / total if total > 0 else 0
        
        analysis = {
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "clarification_rate": clarification_rate,
            "knowledge_growth": metrics["knowledge_additions"],
            "total_interactions": total,
            "recommendations": [],
        }
        
        # Generate recommendations
        if failure_rate > 0.3:
            analysis["recommendations"].append(
                "High failure rate detected. Consider improving interpretation prompts."
            )
        
        if clarification_rate > 0.2:
            analysis["recommendations"].append(
                "Frequent clarification requests. May need better conflict resolution."
            )
        
        if success_rate > 0.9:
            analysis["recommendations"].append(
                "Excellent performance! System is operating optimally."
            )
        
        self.state["last_analysis"] = datetime.now().isoformat()
        self._save_state()
        
        return analysis
    
    def reflect_on_cognitive_process(
        self,
        recent_interactions: List[Dict[str, Any]],
    ) -> str:
        """Use LLM to reflect on recent cognitive processes.
        
        Args:
            recent_interactions: List of recent interaction records.
            
        Returns:
            A natural language reflection on cognitive processes.
        """
        logger.info("[Metacognitive] Reflecting on cognitive processes...")
        
        if not recent_interactions:
            return "I haven't had enough interactions yet to reflect meaningfully."
        
        # Summarize recent interactions
        summary = self._summarize_interactions(recent_interactions)
        
        system_prompt = (
            "You are a metacognitive analysis engine. Your task is to reflect on "
            "recent cognitive processes and identify patterns, strengths, and areas "
            "for improvement. Be specific and actionable."
        )
        
        prompt = (
            f"Recent Cognitive Activity Summary:\n{summary}\n\n"
            "Please provide a brief reflection on:\n"
            "1. What patterns do you notice in the cognitive processes?\n"
            "2. What seems to be working well?\n"
            "3. What could be improved?\n"
            "4. Any insights about learning or reasoning effectiveness?"
        )
        
        try:
            llm_config = self.config.copy()
            llm_config["temperature"] = 0.7
            llm_config["max_tokens"] = 512
            
            reflection = get_response(
                prompt=prompt,
                config=llm_config,
                system_context=system_prompt,
            )
            
            if reflection:
                logger.info("[Metacognitive] Generated reflection on cognitive processes.")
                return reflection
            
            return "I attempted to reflect on my cognitive processes but couldn't generate insights."
            
        except Exception as e:
            logger.error(f"[Metacognitive] Reflection failed: {e}")
            return "I encountered an error while reflecting on my cognitive processes."
    
    def _summarize_interactions(self, interactions: List[Dict[str, Any]]) -> str:
        """Summarize recent interactions for analysis."""
        if not interactions:
            return "No recent interactions."
        
        summary_parts = []
        for i, interaction in enumerate(interactions[-10:], 1):  # Last 10
            interaction_type = interaction.get("type", "unknown")
            success = interaction.get("success", False)
            summary_parts.append(
                f"{i}. {interaction_type}: {'Success' if success else 'Failed'}"
            )
        
        return "\n".join(summary_parts)
    
    def optimize_parameters(self) -> Dict[str, Any]:
        """Analyze performance and suggest parameter optimizations.
        
        Returns:
            A dictionary of suggested parameter changes.
        """
        logger.info("[Metacognitive] Optimizing parameters...")
        
        analysis = self.analyze_performance()
        optimizations = self.state["optimizations"].copy()
        changes = {}
        
        # Adjust confidence threshold based on success rate
        success_rate = analysis.get("success_rate", 0.5)
        current_threshold = optimizations["confidence_threshold"]
        
        if success_rate > 0.9:
            # High success - can be more confident
            new_threshold = min(0.9, current_threshold + 0.05)
            if new_threshold != current_threshold:
                optimizations["confidence_threshold"] = new_threshold
                changes["confidence_threshold"] = new_threshold
        elif success_rate < 0.7:
            # Lower success - be more cautious
            new_threshold = max(0.5, current_threshold - 0.05)
            if new_threshold != current_threshold:
                optimizations["confidence_threshold"] = new_threshold
                changes["confidence_threshold"] = new_threshold
        
        # Adjust interpretation temperature based on clarification rate
        clarification_rate = analysis.get("clarification_rate", 0.0)
        current_temp = optimizations["interpretation_temperature"]
        
        if clarification_rate > 0.2:
            # Too many clarifications - be more precise
            new_temp = max(0.0, current_temp - 0.1)
            if new_temp != current_temp:
                optimizations["interpretation_temperature"] = new_temp
                changes["interpretation_temperature"] = new_temp
        
        if changes:
            self.state["optimizations"] = optimizations
            self._save_state()
            logger.info(f"[Metacognitive] Applied optimizations: {changes}")
        
        return {
            "changes": changes,
            "current_parameters": optimizations,
            "reasoning": analysis,
        }
    
    def identify_patterns(self) -> Dict[str, Any]:
        """Identify patterns in interactions and knowledge.
        
        Returns:
            A dictionary of identified patterns.
        """
        logger.info("[Metacognitive] Identifying patterns...")
        
        # Analyze knowledge graph structure
        node_count = len(self.graph.graph.nodes)
        edge_count = len(self.graph.graph.edges)
        
        # Get most connected nodes (core concepts)
        node_degrees = dict(self.graph.graph.degree())
        top_nodes = sorted(
            node_degrees.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        patterns = {
            "knowledge_structure": {
                "total_concepts": node_count,
                "total_relationships": edge_count,
                "average_connections": edge_count / node_count if node_count > 0 else 0,
                "core_concepts": [self.graph.get_node_by_id(node_id).name for node_id, _ in top_nodes if self.graph.get_node_by_id(node_id)],
            },
            "interaction_patterns": self.state["patterns"],
        }
        
        return patterns
    
    def get_self_report(self) -> str:
        """Generate a comprehensive self-report using LLM.
        
        Returns:
            A natural language self-report.
        """
        logger.info("[Metacognitive] Generating self-report...")
        
        analysis = self.analyze_performance()
        patterns = self.identify_patterns()
        optimizations = self.optimize_parameters()
        
        system_prompt = (
            "You are MOTHER's metacognitive analysis system. Generate a clear, "
            "insightful self-report about cognitive performance and patterns."
        )
        
        prompt = (
            f"Performance Analysis:\n{json.dumps(analysis, indent=2)}\n\n"
            f"Identified Patterns:\n{json.dumps(patterns, indent=2)}\n\n"
            f"Recent Optimizations:\n{json.dumps(optimizations, indent=2)}\n\n"
            "Generate a brief, natural language self-report summarizing these findings."
        )
        
        try:
            llm_config = self.config.copy()
            llm_config["temperature"] = 0.7
            llm_config["max_tokens"] = 512
            
            report = get_response(
                prompt=prompt,
                config=llm_config,
                system_context=system_prompt,
            )
            
            if report:
                return report
            
            return "I attempted to generate a self-report but encountered an issue."
            
        except Exception as e:
            logger.error(f"[Metacognitive] Self-report generation failed: {e}")
            return "I couldn't generate a self-report at this time."

