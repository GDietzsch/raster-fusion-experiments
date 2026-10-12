"""Shared services and the numbered algorithm folders."""
from pathlib import Path
_root = Path(__file__).resolve().parents[4]
__path__.extend(str(_root / name) for name in ['01', '02', '03', '04', '05', '06', '07', '08', '08A', '09', '09A', '10', '10A', '10B', '10C', '11', '12'])
