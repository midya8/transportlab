import json
import os
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def test_solve_and_assignment(self):
        with tempfile.TemporaryDirectory() as d:
            f=os.path.join(d,'input.json');v={'source_points':[[0.,0.],[1.,1.]],'target_points':[[.2,.2],[1.2,1.2]],'source_weights':[.5,.5],'target_weights':[.5,.5]}
            with open(f,'w') as h:json.dump(v,h)
            for command in ['solve','assignment']:
                p=subprocess.run([sys.executable,'-m','transportlab',command,f],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr);r=json.loads(p.stdout);self.assertEqual(len(r['plan']),2)
            v['source_weights']=[.7,.3]
            with open(f,'w') as h:json.dump(v,h)
            p=subprocess.run([sys.executable,'-m','transportlab','assignment',f],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertNotIn('Traceback',p.stderr)

    def test_invalid_schema(self):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            json.dump({},f);f.flush()
            p=subprocess.run([sys.executable,'-m','transportlab','solve',f.name],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertNotIn('Traceback',p.stderr)
