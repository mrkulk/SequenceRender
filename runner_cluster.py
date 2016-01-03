#run script using slurm
import os

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")

# Don't give it a `save` name - that gets generated for you
jobs = [
         {
            'num_entities': 10
         },
        {
            'num_entities': 15
         },
        {
            'num_entities': 30
         },
        {
            'num_entities': 45
         },
]

for job in jobs:
    jobname = "sequence_render_"
    flagstring = ""
    for flag in job:
      jobname = jobname + "_" + flag + "_" + str(job[flag])
      flagstring = flagstring + " --" + flag + " " + str(job[flag])
    flagstring = flagstring + " --save slurm_logs/" + jobname

    if not os.path.exists("slurm_logs/" + jobname):
        os.makedirs("slurm_logs/" + jobname)

    with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
        slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
        slurmfile.write("th main.lua" + flagstring)

    print ("th main.lua" + flagstring)
    if False:
        os.system("sbatch -N 1 -c 2 --gres=gpu:1 --time=5-00:00:00 slurm_scripts/" + jobname + ".slurm &")




