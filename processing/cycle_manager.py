"""
Cycle Manager for MOTHER

Manages autonomous learning cycles that run opportunistically when MOTHER is idle.
Implements Discovery, Study, Refinement, and Metacognitive phases.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from apscheduler.schedulers.base import BaseScheduler
    from memory.knowledge_harvester import KnowledgeHarvester
    from processing.metacognitive_engine import MetacognitiveEngine

logger = logging.getLogger(__name__)


class CycleManager:
    """Manages MOTHER's phased cognitive cycles for autonomous learning."""
    
    def __init__(
        self,
        scheduler: "BaseScheduler",
        harvester: "KnowledgeHarvester",
        metacognitive_engine: Optional["MetacognitiveEngine"] = None,
        config: dict = None,
    ):
        """Initialize the cycle manager.
        
        Args:
            scheduler: APScheduler instance for scheduling cycles.
            harvester: KnowledgeHarvester instance for learning cycles.
            metacognitive_engine: Optional MetacognitiveEngine for self-improvement.
            config: Configuration dictionary.
        """
        self.scheduler = scheduler
        self.harvester = harvester
        self.metacognitive_engine = metacognitive_engine
        self.config = config or {}
        
        # Phase durations (configurable)
        self.LEARNING_PHASE_DURATION = timedelta(
            minutes=self.config.get("learning_phase_duration_minutes", 30)
        )
        self.DISCOVERY_PHASE_DURATION = timedelta(
            minutes=self.config.get("discovery_phase_duration_minutes", 10)
        )
        self.REFINEMENT_PHASE_DURATION = timedelta(
            minutes=self.config.get("refinement_phase_duration_minutes", 5)
        )
        self.METACOGNITIVE_PHASE_DURATION = timedelta(
            minutes=self.config.get("metacognitive_phase_duration_minutes", 5)
        )
        
        # Idle detection
        self.last_user_interaction: Optional[datetime] = None
        self.idle_threshold_minutes = self.config.get("idle_threshold_minutes", 5)
        self.is_idle = False
        self.learning_enabled = self.config.get("enable_autonomous_learning", True)
        
        # Phase tracking
        self.current_phase: Optional[str] = None
        self.phase_start_time: Optional[datetime] = None
        self.cycle_lock = threading.Lock()
        
        # Learning goals (can be set externally)
        self.learning_goals: list = []
    
    def update_user_interaction(self):
        """Call this whenever a user interacts with MOTHER."""
        with self.cycle_lock:
            self.last_user_interaction = datetime.now()
            self.is_idle = False
            # Pause cycles when user is active
            self._pause_cycles()
    
    def _is_idle(self) -> bool:
        """Check if MOTHER is currently idle (no user interaction for threshold)."""
        if not self.last_user_interaction:
            return True  # Never interacted, consider idle
        
        idle_duration = datetime.now() - self.last_user_interaction
        return idle_duration >= timedelta(minutes=self.idle_threshold_minutes)
    
    def start(self) -> None:
        """Start the cycle manager and begin opportunistic learning."""
        if not self.learning_enabled:
            logger.info("[Cycle Manager]: Autonomous learning is disabled")
            return
        
        logger.info("[Cycle Manager]: Starting opportunistic learning cycles")
        
        # Start with learning phase
        self._start_learning_phase()
        
        # Schedule phase management (checks every minute)
        self.scheduler.add_job(
            self._manage_phases,
            "interval",
            minutes=1,
            id="cycle_manager_job",
            replace_existing=True,
        )
    
    def _manage_phases(self) -> None:
        """Main heartbeat - checks for phase transitions and idle state."""
        # Check if user is active
        if not self._is_idle():
            if self.is_idle:
                # User just returned, pause cycles
                self.is_idle = False
                self._pause_cycles()
            return
        
        # User is idle, resume cycles if paused
        if not self.is_idle:
            self.is_idle = True
            logger.info("[Cycle Manager]: MOTHER is idle, resuming learning cycles")
            if not self.current_phase:
                self._start_learning_phase()
        
        # Manage phase transitions
        if not self.phase_start_time:
            return
        
        now = datetime.now()
        elapsed_time = now - self.phase_start_time
        
        if (
            self.current_phase == "learning"
            and elapsed_time >= self.LEARNING_PHASE_DURATION
        ):
            self._start_discovery_phase()
        elif (
            self.current_phase == "discovery"
            and elapsed_time >= self.DISCOVERY_PHASE_DURATION
        ):
            self._start_refinement_phase()
        elif (
            self.current_phase == "refinement"
            and elapsed_time >= self.REFINEMENT_PHASE_DURATION
        ):
            if self.metacognitive_engine:
                self._start_metacognitive_phase()
            else:
                # Skip metacognitive, restart learning
                self._start_learning_phase()
        elif (
            self.current_phase == "metacognitive"
            and elapsed_time >= self.METACOGNITIVE_PHASE_DURATION
        ):
            self._start_learning_phase()
    
    def _pause_cycles(self) -> None:
        """Pause all learning cycles when user is active."""
        self._clear_all_jobs()
        self.current_phase = None
        self.phase_start_time = None
        logger.debug("[Cycle Manager]: Paused cycles (user active)")
    
    def _clear_all_jobs(self) -> None:
        """Remove all existing cognitive cycle jobs from the scheduler."""
        for job in self.scheduler.get_jobs():
            if job.id not in ["cycle_manager_job"]:
                try:
                    job.remove()
                except Exception:
                    pass
    
    def _start_learning_phase(self) -> None:
        """Configure the scheduler to run the Learning Phase cycles."""
        if not self.is_idle:
            return  # Don't start if user is active
        
        logger.info("[Cycle Manager]: Starting LEARNING phase")
        self._clear_all_jobs()
        
        # Run study cycle every 2 minutes during learning phase
        self.scheduler.add_job(
            self._run_study_cycle,
            "interval",
            minutes=2,
            id="study_cycle_job",
            replace_existing=True,
        )
        
        self.current_phase = "learning"
        self.phase_start_time = datetime.now()
    
    def _start_discovery_phase(self) -> None:
        """Configure the scheduler to run the Discovery Phase cycles."""
        if not self.is_idle:
            return
        
        logger.info("[Cycle Manager]: Starting DISCOVERY phase")
        self._clear_all_jobs()
        
        # Run discovery cycle every 5 minutes
        self.scheduler.add_job(
            self._run_discovery_cycle,
            "interval",
            minutes=5,
            id="discover_cycle_job",
            replace_existing=True,
        )
        
        self.current_phase = "discovery"
        self.phase_start_time = datetime.now()
    
    def _start_refinement_phase(self) -> None:
        """Configure the scheduler to run the Refinement Phase cycles."""
        if not self.is_idle:
            return
        
        logger.info("[Cycle Manager]: Starting REFINEMENT phase")
        self._clear_all_jobs()
        
        # Run refinement cycle every 2 minutes
        self.scheduler.add_job(
            self._run_refinement_cycle,
            "interval",
            minutes=2,
            id="refinement_cycle_job",
            replace_existing=True,
        )
        
        self.current_phase = "refinement"
        self.phase_start_time = datetime.now()
    
    def _start_metacognitive_phase(self) -> None:
        """Configure the scheduler to run the Metacognitive Phase cycles."""
        if not self.is_idle:
            return
        
        if not self.metacognitive_engine:
            # Skip to learning phase
            self._start_learning_phase()
            return
        
        logger.info("[Cycle Manager]: Starting METACOGNITIVE phase")
        self._clear_all_jobs()
        
        # Run metacognitive cycle every 2 minutes
        self.scheduler.add_job(
            self._run_metacognitive_cycle,
            "interval",
            minutes=2,
            id="metacognitive_cycle_job",
            replace_existing=True,
        )
        
        self.current_phase = "metacognitive"
        self.phase_start_time = datetime.now()
    
    def _run_study_cycle(self) -> None:
        """Wrapper to run study cycle (checks idle state)."""
        if not self.is_idle:
            return
        try:
            self.harvester.study_cycle(learning_goals=self.learning_goals)
        except Exception as e:
            logger.error(f"[Cycle Manager]: Study cycle error: {e}")
    
    def _run_discovery_cycle(self) -> None:
        """Wrapper to run discovery cycle (checks idle state)."""
        if not self.is_idle:
            return
        try:
            self.harvester.discover_cycle()
        except Exception as e:
            logger.error(f"[Cycle Manager]: Discovery cycle error: {e}")
    
    def _run_refinement_cycle(self) -> None:
        """Wrapper to run refinement cycle (checks idle state)."""
        if not self.is_idle:
            return
        try:
            self.harvester.refinement_cycle()
        except Exception as e:
            logger.error(f"[Cycle Manager]: Refinement cycle error: {e}")
    
    def _run_metacognitive_cycle(self) -> None:
        """Wrapper to run metacognitive cycle (checks idle state)."""
        if not self.is_idle:
            return
        if not self.metacognitive_engine:
            return
        try:
            self.metacognitive_engine.run_introspection_cycle()
        except Exception as e:
            logger.error(f"[Cycle Manager]: Metacognitive cycle error: {e}")
    
    def add_learning_goal(self, goal: str) -> None:
        """Add a learning goal for the study cycle to work on.
        
        Args:
            goal: The concept or topic to learn about.
        """
        if goal not in self.learning_goals:
            self.learning_goals.append(goal)
            logger.info(f"[Cycle Manager]: Added learning goal: '{goal}'")
    
    def get_learning_goals(self) -> list:
        """Get the current list of learning goals."""
        return self.learning_goals.copy()

