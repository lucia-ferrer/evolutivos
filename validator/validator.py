import json
import numpy as np
import os
import signal
import subprocess
import sys

INP_FILE = 'ulises.npy'
TIMEOUT = 10 # seconds


def length (M, route):
    """
    Calculates the length of a route.

    @param M: the matrix with the distances.
    @param route: the route to calculate.
    @return the length of the route.
    """
    return sum([M[route[i], route[(i+1) % len(route)]] for i in range(len(M))])


def test (cmd, file, M):
    """
    Tests the output of a GA command.

    @param cmd: the command to evaluate.
    @param file: the datafile to introduce.
    @param M: the matrix with the distances.
    @return a score of the tested command.
    """
    proc = subprocess.Popen([sys.executable, cmd, file], stdout=subprocess.PIPE)
    try:
        proc.wait(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        output, _ = proc.communicate()
    route = json.loads(output.decode('utf-8'))
    return length(M, route)


if __name__ == '__main__':
    M = np.load('ulises.npy')
    for folder in filter(lambda f : os.path.isdir(f), os.listdir('.')):
        try:
            score = test(f'{folder}/main.py', INP_FILE, M)
            print(f'{folder}: {score}')
        except Exception as e:
            print(f'Ha ocurrido un error con {folder}: {e}')