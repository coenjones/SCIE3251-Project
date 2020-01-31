from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os
import operator
import csv
import time

passnumbersdict = {}
impY = {}
impX = {}
a = []
mapping = []
nodeX = []
nodeY = []

fname = 'BrisbaneApril2019.csv'
fpath = os.path.join('datasets',fname)
with open(fpath,'r') as f:
    reader = csv.reader(f)
    for entry in reader:
        #entry = line.strip().split(',')'Transport for Brisbane' and len(passnumbersdict) < 1000:
        if entry[0] == 'Queensland Rail':
            if entry[6] in passnumbersdict:
                passnumbersdict[entry[6]] += int(entry[8])
            elif (len(passnumbersdict)) < 3000:
                passnumbersdict[entry[6]] = int(entry[8])
f.close()
print(len(passnumbersdict), 'stations')
count = 0
for k in passnumbersdict:
    count += passnumbersdict[k]
print(count/len(passnumbersdict), 'average number pass')


# Imports stop names for cross referencing
stops = {}
fname = 'stops.txt'
fpath = os.path.join('datasets',fname)
with open(fpath,'r') as f:
    reader = csv.reader(f)
    for line in reader:
        stops[line[0]] = line[2]


for k in passnumbersdict:
    a.append(passnumbersdict[k])
    mapping.append(k)
    
    

# Set-Up
NumLines = int(input("How many lines would you like?  "))
NumNodes = len(a)

N = range(NumNodes)
L = range(NumLines)

b = [([1]*NumNodes) for _ in range(NumNodes)] # Congestion Vector
m = [] # Vector of number of stations on each line
delta = [] # NxL matrix of if node n goes on line L
T = int(input("Set the poisson coefficient for the length of lines:  ")) # Poisson Coeffcient regarding length of lines
c = [([0]*NumNodes) for _ in range(NumNodes)] # NxN vector of number of lines running between i and j at the current point in time
W = [] # Stations along each line in order
s = [([2]*len(N)) for _ in N] # Number of transfers required from node i to j
linesthru = [([]) for _ in N]
connects2 = [([0]*len(L)) for _ in L]
whobuilt = [] # List of which player built which line
avpasslin = [] # List of average passengers per line

# Generate distances between nodes and the 'dummy node' to create this a one-way shortest path
# =============================================================================
# distances = [[distance.euclidean([nodeX[k],nodeY[k]],[nodeX[j],nodeY[j]]) for k in N] for j in N]
# distances.append([0]*(len(distances[0]) + 1))
# for k in range(len(distances) - 1):
#     distances[k].append(0)
#     
# =============================================================================
###################################

# Generating delta
for k in N:
    line = [0]*NumLines
    delta.append(line)
    
# Generating Line Lengths
m = [(np.random.poisson(T) + 2) for k in L]


# Main Set-Up Problem Loop
# Operating so as to maximise the compnay benefit
start = time.time()
for k in L:
    line = []
    f = int(np.random.uniform(0,NumNodes))
    delta[f][k] = 1
    line.append(f)
    options = [0]*NumNodes
    while len(line) < m[k]:
        numlinesthru = [sum(k) for k in delta]
       for count in N:
            actsizeatcount = [delta[count][lin]*m[lin] for lin in L]
            actsizeatcount = sum(actsizeatcount)
            g = numlinesthru[count]*actsizeatcount
            g = max(g,1)
            options[count] = a[count]/g
            if count in line:
                options[count] = 0
        for count in N:
            if options[count] == max(options):
                delta[count][-1] = 1
                if len(line) > 1:
                    c[count][line[-1]] += 1
                    c[line[-1]][count] += 1
                line.append(count)
                numlinesthru[count] += 1
    #distancesubset = [[distances[one][two] for one in (line + [len(N)])] for two in (line + [len(N)])]
    W.append(line)
    whobuilt.append(0)
    linepass = [a[k] for k in line]
    avpasslin.append((sum(linepass)/len(linepass)))
end = time.time()
print('Set-up took', round(end-start,3), 'seconds')

# =============================================================================
# # Plotting the setup phase
# plt.plot(nodeX,nodeY,'ro')
# 
# types = ['k-','b-','g-','m-','y-','c-']
# 
# for lin in L:
#     WX = []
#     WY = []
#     for count in range(len(W[lin])):
#         WX.append(nodeX[W[lin][count]])
#         WY.append(nodeY[W[lin][count]])
#     plt.plot(WX[0:len(W[lin])], WY[0:len(W[lin])], types[lin], lin)
#     
# plt.title("Setup Map")
# plt.legend()
# plt.show()
# 
# =============================================================================

input("Press enter to go through to the game stage")
EndOfTime = int(input("How long would you like the game to last for?  "))
Time = range(EndOfTime)


            
# Setup Linesthru            
for A in N:
    for B in L:
        if delta[A][B] == 1:
            linesthru[A].append(B)

# Setup Connects2
for A in N:
    if len(linesthru[A]) > 1:
        for combo in permutations(linesthru[A],2):
            connects2[combo[0]][combo[1]] = 1
            connects2[combo[1]][combo[0]] = 1


# Set 1 transfers to 1
for lin1 in L:
    for lin2 in L:
        for A in W[lin1]:
            for B in W[lin2]:
                if (not A in W[lin2]) and (not B in W[lin1]) and (lin1 != lin2) and connects2[lin1][lin2] == 1:
                    s[A][B] = 1
                    s[B][A] = 1

            
# Set 0 transfers to 0
for nod1 in N:
    for nod2 in N:
        for lin1 in L:
            if (N[nod1] in W[lin1]) and (N[nod2] in W[lin1]):
                s[nod1][nod2] = 0
                s[nod2][nod1] = 0
                
# Game Loop
benefits = [[],[],[]]
for t in Time:
    # Evolutions
    start = time.time()
    for k in range(len(a)):
        a[k] = a[k]*(1-(sum(delta[k])/sum(sum(delta,[])))) # Evolution of passengers at stations
    end=time.time()
    print('Passengers took',round(end-start,3),'seconds')
# =============================================================================
#     start = time.time()
#     for k in range(NumNodes):
#         for l in range(NumNodes):
#             b[k][l] = b[k][l]*(1+(c[k][l]/sum(sum(c,[])))) # Evolution of Congestion
#     # Calculating benefits and choosing players
#     end=time.time()
#     print('Congestion took',round(end-start,3),'seconds')
# =============================================================================
    start=time.time()
    servedpass = [a[k]*(numlinesthru[k] > 0) for k in N]
    servedstat = [(numlinesthru[k] > 0) for k in N]
    served = sum(servedpass)/sum(servedstat)
    totes = sum(a)/len(N)
    avga = served/totes
    avgs = (sum([sum(k) for k in s])/(len(N)**2)) + 1
    avgt = sum(m)/len(L)
    avgh = sum(numlinesthru)/len(N)
    compben = avga*avgs/avgt
    passben = avgh/avgs
    govtben = 1/(avga*avgh*avgs)
    benefits[0].append(compben)
    benefits[1].append(passben)
    benefits[2].append(govtben)
    print('At turn ',t+1, 'Benefits are:')
    print('Company: ', compben)
    print('Passenger: ', passben)
    print('Government: ', govtben)
    total = 2*(compben+passben+govtben)
    compbar = passben+govtben
    passbar = compbar+compben+govtben
    spin = np.random.uniform(0,total)
    end=time.time()
    print('benefits took', round(end-start,3), 'seconds')
    # Calculation to add line
    start=time.time()
    L = range(len(L) + 1)
    m.append(np.random.poisson(T) + 2)
    line = []
    for k in N:
        delta[k].append(0)
    if spin <= compbar: # Company plays
        player = 0
        options = [0]*NumNodes
        while len(line) < m[-1]:
            for count in N:
                actsizeatcount = [delta[count][lin]*m[lin] for lin in L]
                actsizeatcount = sum(actsizeatcount)
                g = numlinesthru[count]*actsizeatcount
                g = max(g,1)
                options[count] = a[count]/g
                if count in line:
                    options[count] = 0
            for count in N:
                if options[count] == max(options):
                    delta[count][-1] = 1
                    if len(line) > 1:
                        c[count][line[-1]] += 1
                        c[line[-1]][count] += 1
                    line.append(count)
                    numlinesthru[count] += 1
        while len(line) > m[-1]:
            line.pop(-1)
        W.append(line)
        whobuilt.append(0)
        linepass = [a[k] for k in line]
        avpasslin.append((sum(linepass)/len(linepass)))
    elif spin <= passbar: # Passenger plays
        player = 1
        options = [0]*NumNodes
        while len(line) < m[-1]:
            for count in N:
                actsizeatcount = [delta[count][lin]*m[lin] for lin in L]
                actsizeatcount = sum(actsizeatcount)
                g = numlinesthru[count]*actsizeatcount
                g = max(g,1)
                options[count] = a[count]/g
                if count in line:
                    options[count] = 1000000
            for count in N:
                if options[count] == min(options):
                    delta[count][-1] = 1
                    if len(line) > 1:
                        c[count][line[-1]] += 1
                        c[line[-1]][count] += 1
                    line.append(count)
                    numlinesthru[count] += 1
        while len(line) > m[-1]:
            line.pop(-1)
        W.append(line)
        whobuilt.append(1)
        linepass = [a[k] for k in line]
        avpasslin.append((sum(linepass)/len(linepass)))
    else: # Government plays
        player = 2
        options = [0]*NumNodes
        while len(line) < m[-1]:
            for count in N:
                actsizeatcount = [delta[count][lin]*m[lin] for lin in L]
                actsizeatcount = sum(actsizeatcount)
                g = numlinesthru[count]*actsizeatcount
                options[count] = a[count]*g
                if count in line:
                    options[count] = 1000000
            for count in N:
                if options[count] == min(options):
                    delta[count][-1] = 1
                    if len(line) > 1:
                        c[count][line[-1]] += 1
                        c[line[-1]][count] += 1
                    line.append(count)
                    numlinesthru[count] += 1
        while len(line) > m[-1]:
            line.pop(-1)
        W.append(line)
        whobuilt.append(2)
        linepass = [a[k] for k in line]
        avpasslin.append((sum(linepass)/len(linepass)))
    end = time.time()
    print('Player Number: ', player, 'has been chosen, and they took', round(end-start,3),'seconds')

    start = time.time()
    # Update linesthru, connects2, s        
    linesthru = [([]) for _ in N]
    for A in N:
        for B in L:
            if delta[A][B] == 1:
                linesthru[A].append(B)
    connects2 = [([0]*len(L)) for _ in L]
    for A in N:
        if len(linesthru[A]) > 1:
            for combo in permutations(linesthru[A],2):
                connects2[combo[0]][combo[1]] = 1
                connects2[combo[1]][combo[0]] = 1
    for lin1 in L:
        for lin2 in L:
            for A in W[lin1]:
                for B in W[lin2]:
                    if (not A in W[lin2]) and (not B in W[lin1]) and (lin1 != lin2) and connects2[lin1][lin2] == 1:
                        s[A][B] = 1
                        s[B][A] = 1
    for nod1 in N:
        for nod2 in N:
            for lin1 in L:
                if (N[nod1] in W[lin1]) and (N[nod2] in W[lin1]):
                    s[nod1][nod2] = 0
                    s[nod2][nod1] = 0
    end = time.time()
    print('Updating transfers etc. took', round(end-start,3), 'seconds')      
# Plot the benefits
"""
plt.plot(Time,benefits[0],'g^-')
plt.plot(Time,benefits[1],'ro-')
plt.plot(Time,benefits[2],'ks-')
plt.xlabel('Turn Number')
plt.ylabel('Benefit')
plt.legend(['Company','Passenger','Government'])
plt.show()
"""

# See the station names on each line
"""
 for station in W:
    print(stops[mapping[station]])
"""

# Get number of passengers for a line
"""
linepass = [a[k] for k in W[0]]
avg = sum(linepass)/len(linepass)
"""

# Average passengers per type
"""
ride0 = [avpasslin[m] for m in L if whobuilt[m] == 0]
ride1 = [avpasslin[m] for m in L if whobuilt[m] == 1]
ride2 = [avpasslin[m] for m in L if whobuilt[m] == 2]
avride0 = sum(ride0)/len(ride0)
avride1 = sum(ride1)/len(ride1)
avride2 = sum(ride2)/len(ride2)
plt.bar([0,1,2],[avride0,avride1,avride2],log=True, tick_label = ['Company','Passenger','Government'],color=['g','r','k'])
"""




