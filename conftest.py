"""
Pytest configuration — adds the project root to sys.path so that
`src.*` imports work without needing `pip install -e .`.
"""
import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))
