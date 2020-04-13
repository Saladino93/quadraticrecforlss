import subprocess

import sys

if len(sys.argv) == 1:
        print('Choose your configuration file!')
        sys.exit()

config = str(sys.argv[1])

program_list = ['create_matter_power.py', 'generate.py', 'forecast.py']

for program in program_list:
    subprocess.call(['python3', program, config])
    print("Finished:" + program)
