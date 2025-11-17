"""
TinyNotify-LLM Dashboard
Interactive Streamlit UI for notification system.
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ui.dashboard import main

if __name__ == "__main__":
    main()
