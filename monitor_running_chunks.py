import os
import json
import pandas as pd
import time

os.system("rm in_queue.json")
os.system("squeue --json >> in_queue.json")

with open('in_queue.json', 'r') as file:
    data = json.load(file)

useful_tags = ["job_id", "allocating_node","job_state"]

jobs = data["jobs"]

slurm_dir = "/cluster/home/ccarissimo/braessQlearning/"

# print(jobs[0])

start = 1737732301000/10**3
now = time.time()
elapsed = (now - start)/(60*60)
total_jobs = 1000
print(f"Jobs: {total_jobs - len(jobs)} out of {total_jobs} completed")
estimated_time_remaining = (len(jobs) * elapsed/(total_jobs - len(jobs))) if len(jobs)!=total_jobs else 0
print(f"time elapsed: {elapsed} hours")
print(f"estimated time remaining: {estimated_time_remaining} hours, equal to {estimated_time_remaining/24} days")

frames = []
for job in jobs:
    row = {key: job[key] for key in useful_tags}
    # row["time"] = (row["accrue_time"] - time.time())/60000
    try:
        with open(f'{slurm_dir}/slurm-{job["job_id"]}.out', 'r') as f:
            last_line = f.readlines()[-1]
    except Exception as error:
        last_line = "pending"
    row["progress_bar"] = last_line
    if job["job_state"] != "PENDING":
        row["cores"] = job["job_resources"]["allocated_cores"]
        frames.append(row)

    # os.system(f"scancel {job['job_id']}")

df = pd.DataFrame(frames)

print(df.to_markdown())

df.to_csv(slurm_dir + "in_queue.csv")
