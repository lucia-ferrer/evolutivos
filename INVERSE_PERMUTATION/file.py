def encode(perm):
    """
    input : permutation i0...iN
    output: inverse permutation a0 ... aN
    For a permutation i1, i2, . . . , iN of the set {1, 2, . . . , N } we let aj denote the number of integers in the permutation which precede j but are greater than j. 
    So, aj is a measure of how much out of order j is. 
    The sequence of numbers a1, a2, . . . , aN is called the inversion sequence of the permutation i1, i2, . . . , iN . 
    The inversion sequence a1, a2, . . . , aN satisfies the conditions 0 ≤ ai ≤ N − i for i = 1, 2, . . . , N
    As seen there is no restriction on the elements which says ai = aj is forbidden for i different of j.
    This is of course very convenient for the crossover and mutation operations in GA.
    """
    N = len(perm)
    inv = [0] * N  # Initialize the inversion sequence with zeros

    for i in range(N):
        inv_i = 0
        m = 1
        while perm[m - 1] != (i + 1):
            if perm[m - 1] > (i + 1):
                inv_i += 1
            m += 1
        inv[i] = inv_i

    return inv

def encode_I(perm):
    """
    input: permutation i0...iN
    output: inverse permutation a0 ... aN
    For a permutation i1, i2, . . . , iN of the set {1, 2, . . . , N } we let aj denote the number of integers in the permutation which precede j but are greater than j. 
    So, aj is a measure of how much out of order j is. 
    The sequence of numbers a1, a2, . . . , aN is called the inversion sequence of the permutation i1, i2, . . . , iN . 
    The inversion sequence a1, a2, . . . , aN satisfies the conditions 0 ≤ ai ≤ N − i for i = 1, 2, . . . , N
    As seen there is no restriction on the elements which says ai = aj is forbidden for i different of j.
    This is, of course, very convenient for the crossover and mutation operations in GA.
    """
    N = len(perm)
    inv = [0] * N  # Initialize the inversion sequence with zeros

    for i in range(N):
        inv_i = 0
        m = 1
        while m <= N and perm[m - 1] != (i + 1):
            if perm[m - 1] > (i + 1):
                inv_i += 1
            m += 1
        inv[i] = inv_i

    return inv

def encode_II(permutation):
    N = len(permutation)
    inversion_sequence = [0] * N

    for i in range(N):
        inv_i = 0
        for j in range(i + 1, N):
            if permutation[j] < permutation[i]:
                inv_i += 1
        inversion_sequence[i] = inv_i

    return inversion_sequence

def decode(inv):
    """
    does the inverse of the method encodr, such that 
    given the inversed permutation a0...aN, it returns the original permutation i0...iN.
    """
    N = len(inv)
    permutation = [0] * N

    for i in range(N):
        count = inv[i]
        j = 0
        while count > 0 or permutation[j] != 0:
            if permutation[j] == 0:
                count -= 1
            j += 1
        permutation[j] = i + 1
    return permutation    

def decode_II(inversion_sequence):
    N = len(inversion_sequence)
    permutation = [0] * N

    for i in range(N):
        count = inversion_sequence[i]
        j = 0
        while count > 0:
            if permutation[j] == 0:
                count -= 1
            j += 1
        permutation[j] = i + 1

    return permutation

def decode_I(inv):
    N = len(inv)
    permutation = [0] * N

    for i in range(N):
        count = inv[i]
        j = 0
        while count > 0 or permutation[j] != 0:
            if permutation[j] == 0:
                count -= 1
            j += 1
            if j == N:  # Check if j exceeds the bounds of the permutation
                break  # Exit the loop to prevent an error
        if j < N:  # Ensure j is within bounds
            permutation[j] = i + 1

    return permutation

def decode_II(inversion_sequence):
    N = len(inversion_sequence)
    permutation = [0] * N

    for i in range(N):
        count = inversion_sequence[i]
        j = 0
        while count > 0:
            if permutation[j] == 0:
                count -= 1
            j += 1
        permutation[j] = i + 1

    return permutation  

def decode_III(inv):
    N = len(inv)
    original = [0] * N
    used = [False] * N

    for i in range(N):
        count = inv[i]
        j = 0
        while count > 0:
            if not used[j]:
                count -= 1
            j += 1

        while used[j]:
            j += 1

        used[j] = True
        original[j] = i + 1

    return original

import random
# Example usage:
perm = [i for i in range(1, 15)]             
random.shuffle(perm) #perm i0, i1,..., iN of the set {0, 1, ..., N }
perm = [15, 10, 1, 4, 8, 13, 5, 7, 14, 3, 11, 16, 12, 9, 6, 2]
print(perm)
print()
inv = encode(perm)
print(inv)
print(decode(inv))

print('method I')
inv = encode_I(perm)
print(inv)
print(decode_I(inv))

print('method II')
inv = encode_II(perm)
print(inv)
print(decode_II(inv))


print('method III')
inv = encode(perm)
print(inv)
print(decode_III(inv))


e = [1, 2, 6, 1, 11, 5, 6, 3, 6, 2, 4, 11, 1, 0, 0, 0]
print('ERROR ', e)

print('METHOD III')
oit = decode_III(e)
print(oit)
print(enconde(oit))


ori = decode_I(e)
pre = encode_I(ori)
print(ori)
print(pre)

print('METHOD II')
oit = decode_II(e)
print(oit)
print(enconde_II(oit))


            #RandGene =>  randint: gLow <= N <= gHi (genera un gen : NUMERO A INSERTAR EN GENOMA)
            #randGenes => genera lista de genes aleatorios tamaño N. 
            #randChrom => genera un cromosoma llamando a randGenes con input gn.
            #randChroms => genera lista de cromosomas de tamaño cn.
            #randPop => caso especial de randChroms pero con cn=numPop, y gn=numVar
            #randIdx => devuelve un index de la lista xs que tiene longitud n, en el intervalo [n-1]


