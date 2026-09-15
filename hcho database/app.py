import os
import sys

# Forward to root unified dashboard
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
unified_script = os.path.join(root_dir, "unified_dashboard.py")

with open(unified_script, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, unified_script, 'exec'))
