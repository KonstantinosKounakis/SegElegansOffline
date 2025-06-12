#!/usr/bin/python3

import subprocess
import os

# Body network assembly

checkpoint_SEG = os.path.join('Models','Body','SEG','model.pth')
checkpoint_SKL = os.path.join('Models','Body','SKL','model.pth')

subprocess.run(["file_split_merge","-m","-i", checkpoint_SEG])
subprocess.run(["file_split_merge","-m","-i", checkpoint_SKL])

