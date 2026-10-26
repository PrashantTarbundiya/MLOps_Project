import os
import sys

# Ensure prediction_model package is discoverable
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

__version__ = "1.0.0"
