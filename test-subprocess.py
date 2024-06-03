import subprocess

import os


class TestSubprocess:
    def __init__(self):
        self.process = None
        self.test_subprocess()
    
    def test_subprocess(self):
        self.process = subprocess.Popen(["mpirun", "--allow-run-as-root", "-n", "4", "/usr/bin/python3", "/root/open-moe-llm-leaderboard/src/backend/trt_runner.py", "--checkpoint", "mistralai/mixtral-8x7b-instruct-v0.1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
    def __del__(self):
        if self.process is not None:
            out, err = self.process.communicate()
            print(f"out: {out}, err: {err}")
            
            
if __name__ == "__main__":
    test = TestSubprocess()
    
    import time
    time.sleep(20)
    del test