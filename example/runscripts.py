import subprocess

import sys

if len(sys.argv) == 1:
        print('Choose your directory!') 
        sys.exit() 
    
direc = str(sys.argv[1])

program_list = ['generate.py', 'forecast.py']

for program in program_list:
    subprocess.call(['python', program, direc])
    print("Finished:" + program)
