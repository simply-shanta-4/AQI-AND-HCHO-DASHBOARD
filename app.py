import os
import sys

# Launch Unified Dashboard
unified_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unified_dashboard.py")
with open(unified_script, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, unified_script, 'exec'))
