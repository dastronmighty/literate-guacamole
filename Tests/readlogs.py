import os
import re

path = "./logs"

# reading = "xor"
# reading = "Sin"
reading = "Letter"

startex = re.compile("[a-zA-Z]+")
numtex = re.compile("[a-zA-Z]+(\d+)")

logs = [_ for _ in os.listdir(path) if reading in _]
name = startex.findall(logs[0])[0]
logs = [int(numtex.findall(_)[0]) for _ in logs]
logs.sort()
logs = [f"{path}/{name}{str(_)}.txt" for _ in logs]

for log in logs:
    print(log)
    with open(log) as f:
        lines = f.read().split("\n")
        print(lines[-3])
        print(lines[-2])
        print(lines[-1])
