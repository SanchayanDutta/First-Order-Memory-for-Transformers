import subprocess, pathlib, sys, tempfile
root = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as td:
    out = pathlib.Path(td) / 'smoke'
    cmd = [sys.executable, str(root/'experiments'/'cache_frontier.py'), '--smoke_test', '--cpu', '--seeds', '0', '--out_dir', str(out)]
    subprocess.check_call(cmd, cwd=str(root))
    assert (out/'summary.csv').exists(), 'summary.csv missing'
    assert (out/'paired_wins.csv').exists(), 'paired_wins.csv missing'
print('smoke test passed')
