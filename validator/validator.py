import json
import numpy as np
import pandas as pd
import os
import signal
import subprocess
import sys
import time
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Ejemplo de programa con argumentos")
    parser.add_argument("file", help="Nombre del archivo (argumento obligatorio)")
    parser.add_argument("-n", type=int, default=60, help="Valor de n (argumento opcional)")
    parser.add_argument("-p", type=float, default=0.3, help="Valor de probabilidad de mutacion (argumento opcional)")
    global args
    args = parser.parse_args()
    return args.file, args.n, args.p

INP_FILE, INP_N, INP_P = get_config()
TIMEOUT =  15000 # seconds ~ 4 h.

def length (M, route):
    """
    Calculates the length of a route.

    @param M: the matrix with the distances.
    @param route: the route to calculate.
    @return the length of the route.
    """
    return sum([M[route[i], route[(i+1) % len(route)]] for i in range(len(M))])


def test (cmd, file, n, p,  M, folder):
    """
    Tests the output of a GA command.

    @param cmd: the command to evaluate.
    @param file: the datafile to introduce.
    @param M: the matrix with the distances.
    @return a score of the tested command.
    """

    #DATA TO STORE: NAME, ARGS, PROBLEM, N, TIME, COST, PATH

    proc = subprocess.Popen([sys.executable, cmd, file, "-n", str(n), "-p", str(p)],stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #record = pd.read_csv("results.csv")
    
    start_time = time.time()
    try:
        proc.wait(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
    end_time = time.time()
    output, _ = proc.communicate()
    route = json.loads(output) #.decode('utf-8')
    results = {"Name":folder, "Args":args, "Problem":INP_FILE, "N":M.shape[0], "Time":end_time-start_time, "Solution":route}
    record = pd.DataFrame(results)
    record.to_csv(os.getcwd() + "/results.csv")
    return route #length(M, route)


if __name__ == '__main__':
    M = np.load(INP_FILE )
    for folder in filter(lambda f : os.path.isdir(f), os.listdir('.')):
        try:
            score = test(f'{folder}/main.py', INP_FILE, INP_N, INP_P, M, folder)
            print(f'{folder}: {score}')
        except Exception as e:
            print(f'Ha ocurrido un error con {folder}: {e}')