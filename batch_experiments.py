import os

for e in np.linspace(0, 2, 11):
    os.system(f'sbatch --time=24:00:00 --wrap="python pipelineExperiments.py {e}')
