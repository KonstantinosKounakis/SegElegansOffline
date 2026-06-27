#!/usr/bin/python3

import subprocess
import os

# Body network assembly

checkpoint_BODY = os.path.join('Models','Body','Checkpoint','model.pth')

subprocess.run(["file_split_merge","-m","-i", checkpoint_BODY])
