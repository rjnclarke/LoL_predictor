"""
viz/base_viz.py
===============

Defines ABCViz — base class for all visualization modules.

Each visualization:
    • Accepts a MatchBase instance (shared DB + metadata)
    • Implements fetch_data() to query required info
    • Implements build_figure() to return a Figure object

By using MatchBase, every visualization automatically has access to:
    • self.conn  → SQLite connection
    • self.mb.level_min / self.mb.level_max for normalization
    • self.mb.log()  → unified logging

This design supports both Matplotlib and Plotly figures and can
later plug directly into Dash callbacks.
"""

from abc import ABC, abstractmethod

class ABCViz(ABC):
    """Abstract base for all visualization modules."""

    def __init__(self, match_base):
        """
        Parameters
        ----------
        match_base : MatchBase
            Active MatchBase instance providing:
                - db.conn  : SQLite connection
                - level_min/max : global metadata
        """
        self.mb = match_base
        self.conn = match_base.db.conn

    # ------------------------- Interface ------------------------- #
    @abstractmethod
    def fetch_data(self, *args, **kwargs):
        """Query and prepare data for visualization."""
        pass

    @abstractmethod
    def build_figure(self, *args, **kwargs):
        """Construct and return a Figure (Matplotlib or Plotly)."""
        pass

    # ------------------------- Utilities -------------------------- #
    def log(self, msg: str):
        """Standard logger shared across all visualizations."""
        self.mb.log(f"[VIZ] {msg}")

    def close(self):
        """Optional cleanup — closes DB connection if needed."""
        try:
            self.conn.close()
        except Exception:
            pass