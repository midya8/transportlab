#!/usr/bin/env bash
set -euo pipefail
installed=${1:?installed package directory required}
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$root"
python -m transportlab demo > "$work/source-demo.json"
python -m transportlab sensitivity examples/direction.json --iterations 3 > "$work/source-sensitivity.json"
cp examples/direction.json "$work/direction.json"
cd "$work"
export PYTHONPATH="$installed"
python -m transportlab demo > installed-demo.json
python -m transportlab sensitivity direction.json --iterations 3 > installed-sensitivity.json
python - <<'PY'
import importlib,pkgutil,json,subprocess,sys
from pathlib import Path
import transportlab
from transportlab.marginal import audit
root=Path(transportlab.__file__).parent
mods=[m for m in pkgutil.walk_packages(transportlab.__path__,transportlab.__name__+'.') if m.name!='transportlab.__main__']
assert len(mods)==6
for module in mods:
    loaded=importlib.import_module(module.name)
    assert Path(loaded.__file__).resolve().is_relative_to(root.resolve())
assert json.load(open('source-demo.json'))==json.load(open('installed-demo.json'))
assert json.load(open('source-sensitivity.json'))==json.load(open('installed-sensitivity.json'))
files=sorted((root/'artifacts').glob('marginal-*.json'));assert len(files)==12
for file in files:
    c=json.loads(file.read_text());assert audit(c['input'])==c['report']
    Path('input.json').write_text(json.dumps(c['input']))
    result=subprocess.run([sys.executable,'-m','transportlab.marginal','input.json'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==c['report']
    bad=dict(c['input'],iterations=501)
    result=subprocess.run([sys.executable,'-m','transportlab.marginal','-'],input=json.dumps(bad),capture_output=True,text=True)
    assert result.returncode==2 and not result.stdout
print('all7importablemodules plus mainCLI from installed package; original complete demo and sensitivity match source;12new controls and12CLI refusals reproduce outside checkout')
PY
