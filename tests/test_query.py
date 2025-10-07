import subprocess, json

def test_cli_runs():
    out = subprocess.check_output(["python", "src/query.py", "space survival"])
    assert b"Top results:" in out
