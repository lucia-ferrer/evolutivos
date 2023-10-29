import numpy as np
import math
def createIndividual():
        fenoma = [1]
        genoma = ""
        solucion = []
        gene_len = int(math.log(m, 2))
        n = distance_matrix.shape[0]
        genes = [i for i in range(n-1, gene_len-1, -1)] 
        
        for i in genes:
            #print(f'fenoma: {fenoma}, solucion: {solucion}, genoma:{genoma}')
            
            c_idx = fenoma[-1]                          # <- la ciudad - 1 = index en la tabla de cercanias. 
            not_visited_nn = [nn_c for nn_c in ordenadas_ids[c_idx-1] if nn_c not in fenoma]
            
            if i > gene_len+1:                       # <- index de 1 a m vecinos cercanos. 
                nn_idx = random.randint(0,m-1)

            elif i <= gene_len+1 : 
                nn_idx = random.randint(0,1)

            try: 
                nn_c = not_visited_nn[nn_idx]               # <- ciudad cercana no Visitada. 
            except IndexError : 
                print(i)
                print(nn_idx)
                print(not_visited_nn)
                quit()
            
            fenoma.append(nn_c)
            solucion.append(nn_idx)
            gene = bin(nn_idx)[2:] 

            if len(gene) < gene_len and i>3:
                # If the binary string is shorter than the desired length, pad with leading zeros
                gene = '0' * (gene_len - len(gene)) + gene

            genoma += gene

        #add the last location in the fenoma but no binary code needed. 
        last_c = [c for c in ordenadas_ids[fenoma[-1]-1] if c not in fenoma]
        fenoma.append(last_c[0])
        return genoma, fenoma, solucion




def decode(individual, m=4):
        """
            Transforma un individuo/genoma en una solución y luego en un camino/fenoma utilizando una matriz de distancias.

            Args:
                individual (str): La representación binaria sub-óptima del individuo, es decir self.genoma.

            Returns:
                tuple: Una tupla que contiene el camino (secuencia de ciudades visitadas) y la solución final.
        """
        genoma = individual
        gene_len = int(math.log(m, 2))
        fenoma = [1]  # Comenzamos desde la ciudad 1
        n = distance_matrix.shape[0]

        #(1) decodificamos los indices de binario para sacar la solucion, indices en decimal. 
        genes_bit = [genoma[i:i+gene_len] for i in range(0,len(genoma)-gene_len, gene_len)]
        print('GenesBIT, ', genes_bit)
        genes_sol = [int(group, gene_len) for group in genes_bit]

        ulti_bit = [genoma[i] for i in range(-gene_len, 0, +1)]
        ulti_sol = [int(bit, gene_len) for bit in ulti_bit]
        
        # print(f'Genoma: {genoma}\nGenesBina: {genes_bit}\nGenesDeci: {genes_sol}\nUltBin: {ulti_bit}\nUltDec: {ulti_sol}')

        solucion = genes_sol + ulti_sol

        #(2) de la solucion sacamos el path.

        for i in range(len(solucion)):
            c_idx = fenoma[-1] - 1  # La ciudad actual en el índice de la tabla de cercanías                    
            not_visited_nn = [nn_c for nn_c in ordenadas_ids[c_idx] if nn_c not in fenoma] # no visitadas
            nn_idx = solucion[i]
            if nn_idx >= len(not_visited_nn):
                nn_c = not_visited_nn[0]  # Si el índice es mayor o igual al número de no visitados, elegimos el primero
            else:
                nn_c = not_visited_nn[nn_idx]

            fenoma.append(nn_c)
        

        # Añadir al final la ciudad no visitada
        not_visited_cities = [c for c in ordenadas_ids[fenoma[-1] - 1] if c not in fenoma]
        fenoma.append(not_visited_cities[0])

        return fenoma, solucion

import random

def single_point_crossover(individual1, individual2):
    """
    Realiza un cruce de un solo punto entre dos individuos en representación binaria sub-óptima.

    Args:
        individual1 (str): Representación binaria sub-óptima del primer individuo.
        individual2 (str): Representación binaria sub-óptima del segundo individuo.

    Returns:
        tuple: Dos descendientes obtenidos del cruce (permutaciones válidas en TSP).
    """
    # Asegúrate de que los individuos tengan la misma longitud
    if len(individual1) != len(individual2):
        raise ValueError("Los individuos deben tener la misma longitud para el cruce de un solo punto.")

    # Elegir un punto de cruce aleatorio
    crossover_point = random.randint(0, len(individual1) - 1)

    # Realizar el cruce
    descendant1 = individual1[:crossover_point] + individual2[crossover_point:]
    descendant2 = individual2[:crossover_point] + individual1[crossover_point:]

    return descendant1, descendant2



def matrix_cercanas_ciudad(adj_matrix):
    n = len(adj_matrix)
    orden_ciudades = np.zeros((n, n), dtype=int)

    for i in range(n):
        # Calcula la distancia de la ciudad i a todas las demás ciudades
        distancias = adj_matrix[i]
        ciudades_posibles = range(1, len(distancias)+1)
        # Enumera las ciudades en función de sus distancias a la ciudad i
        ordenadas = sorted(ciudades_posibles, key=lambda j: distancias[j-1])

        # Almacena los índices de las ciudades ordenadas en la matriz de salida
        orden_ciudades[i] = ordenadas

    # Convierte los índices de las ciudades en nombres de ciudades
    ciudades_ordenadas = []
    
    for i in range(n):
        fila = [ciudades_posibles[j-1] for j in orden_ciudades[i]][1:]
        ciudades_ordenadas.append(fila)
        print(f'ciudad; {i+1}, ciudades_ordenadas:{fila}')
    return ciudades_ordenadas




# Ejemplo de uso:
if __name__ == "__main__":
    # Example distance matrix (replace with your own)
    distance_matrix = np.array([
        [0, 29, 20, 21, 16],
        [29, 0, 15, 29, 28],
        [20, 15, 0, 15, 14],
        [21, 29, 15, 0, 4],
        [16, 28, 14, 4, 0]
    ])

    m = 4  # Replace with your desired "m" value
    ordenadas_ids = matrix_cercanas_ciudad(distance_matrix)
    
    genoma, fenoma, solucion = createIndividual()

    print(f"\nEncoded Binary Representation: Genoma {genoma}, Solucion {solucion}, Fenoma {fenoma}")

    decoded, solution = decode(genoma)
    print(f"Decoded Path (Permutation of Cities): FenomaDec {decoded}, SolDecoded {solution}, Fenoma: {fenoma}\n")


    # Ejemplo de matriz de distancias (debes proporcionar la tuya):
    distance_matrix = np.array([
        [0, 4, 1, 7, 6, 7, 3, 5, 8, 9],
        [4, 0, 6, 4, 5, 8, 8, 9, 5, 7],
        [1, 6, 0, 7, 8, 9, 2, 8, 3, 4],
        [7, 4, 7, 0, 2, 9, 8, 7, 6, 1],
        [6, 5, 8, 2, 0, 8, 9, 3, 4, 7],
        [7, 8, 9, 9, 8, 0, 6, 7, 2, 8],
        [3, 8, 2, 8, 9, 6, 0, 3, 7, 9],
        [5, 9, 8, 7, 3, 7, 3, 0, 8, 6],
        [8, 5, 3, 6, 4, 2, 7, 8, 0, 5],
        [9, 7, 4, 1, 7, 8, 9, 6, 5, 0]
    ])
    
    m = 4  # Replace with your desired "m" value
    ordenadas_ids = matrix_cercanas_ciudad(distance_matrix)

    genoma, fenoma, solucion = createIndividual()

    print(f"\nEncoded Binary Representation: Genoma {genoma}, Solucion {solucion}, Fenoma {fenoma}")

    decoded, solution = decode(genoma)
    print(f"Decoded Path (Permutation of Cities): FenomaDec {decoded}, SolDecoded {solution}, Fenoma: {fenoma}\n")


    # individual1 = Individual(genoma="01010010011011")  # Representación binaria sub-óptima del primer individuo
    # individual2 = Individual(genoma="10100111001011")  # Representación binaria sub-óptima del segundo individuo

    individual1, fenoma, solucion = createIndividual()
    individual2, fenoma, solucion = createIndividual()

    descendant1, descendant2 = single_point_crossover(individual1, individual2)

    print("Individuo 1 (Padre): ", individual1)

    print("Individuo 2 (Madre):", individual2)

    print("\nDescendiente 1:", descendant1)

    print("Descendiente 2:", descendant2)

    path1 = decode(individual1)
    path3 = decode(descendant1)
    path2 = decode(individual2)
    path4 = decode(descendant2)
    print("\nCamino (Secuencia de Ciudades Visitadas):")
    print(path1)
    print(path2)
    print(path3)
    print(path4)


