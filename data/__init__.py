"""
Data module for NQ futures data ingestion, validation, and contract math.
"""
from data.contract import FuturesCalculator
from data.loader import NQDataLoader

__all__ = ["FuturesCalculator", "NQDataLoader"]
