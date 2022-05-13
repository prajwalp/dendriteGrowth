import numba
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L = 51 # use odd number of sites so that central starting point is simple

state = np.zeros((L,L))
state[(L-1)//2,(L-1)//2] = 1

def calculateDendrite(site,lattice):
    x = site[0]
    y = site[1]
    neighbors = []
    if(0<x and x<L-1 and 0<y and y<L-1):
        neighPairs = np.array([[x-1,y-1],[x,y-1],[x+1,y-1],[x-1,y],[x+1,y],[x-1,y+1],[x,y+1],[x+1,y+1]])
        neighStates = lattice[neighPairs[:,0],neighPairs[:,1]]
        zeroSites = np.stack(np.where(neighStates==1),axis=1)
        neighbors = neighPairs[zeroSites]
    return neighbors

def calculateBoundary(site,lattice):
    x = site[0]
    y = site[1]
    neighbors = []
    if(0<x and x<L-1 and 0<y and y<L-1):
        neighPairs = np.array([[x-1,y-1],[x,y-1],[x+1,y-1],[x-1,y],[x+1,y],[x-1,y+1],[x,y+1],[x+1,y+1]])
        neighStates = lattice[neighPairs[:,0],neighPairs[:,1]]
        zeroSites = np.stack(np.where(neighStates==0),axis=1)
        neighbors = neighPairs[zeroSites]
    else:
        neighbors = -1
    return neighbors

@numba.njit
def calculateProb(lattice):
    prob = np.zeros((L+1,L+1))
    prob[0,:] = 1
    prob[L,:] = 1
    prob[:,0] = 1
    prob[:,L] = 1
    
    while(True):
        globalError = 0
        new_prob = prob[:]
        for i in range(L+1):
            for j in range(L+1):                    
                if(i>0 and i<L and j>0 and j<L):
                    if(lattice[i,j] ==1):
                        new_prob[i,j] = 0
                    if(lattice[i,j]!=1):
                        globalError += ((np.sum(prob[i-1:i+2,j-1:j+2])-prob[i,j])/8.0  - prob[i,j])**2
                        new_prob[i,j] = (np.sum(prob[i-1:i+2,j-1:j+2]) - prob[i,j])/8.0
        prob = new_prob[:]
        if(globalError<1e-4*L):
            break
                        
    return prob

temp = calculateProb(state)

state = np.zeros((L,L))
state[(L-1)//2,(L-1)//2] = 1
growthPattern = []

T = 1000

for tIndex in range(T):
    dendriteSites = np.stack(np.where(state==1),axis=1)
    check = 0
    boundary = calculateBoundary(dendriteSites[0],state)
    if(type(boundary)==int):
        print("Reached boundary at time",tIndex)
        break
    for i in range(1,len(dendriteSites)):
        boundarySites = calculateBoundary(dendriteSites[i],state)
        if(type(boundarySites)==int):
            check=-1
            break
        boundary = np.unique(np.concatenate((boundarySites,boundary)),axis=0)
    if(check==-1):
        print("Reached boundary at time",tIndex)
        break
    prob = calculateProb(state)
    nBoundary = len(boundary)
    rates = prob[boundary[:,0,0],boundary[:,0,1]]
    rates = rates/sum(rates)
    check = np.random.random()
    for i in range(len(rates)):
        if(check<sum(rates[:i+1])):
            surroundingDendrites = calculateDendrite([boundary[i][0,0],boundary[i,0,1]],state)
            state[boundary[i][0,0],boundary[i,0,1]] = 1
            if(len(surroundingDendrites)==0):
                break
            randNeigh = np.random.randint(0,len(surroundingDendrites))
            growthPattern.append([surroundingDendrites[randNeigh][0],np.array([boundary[i][0,0],boundary[i,0,1]])])
            break

plt.figure()
finalProb = calculateProb(state)
plt.imshow(finalProb)
plt.colorbar()
plt.savefig("finalProb.png")
plt.close()

growthPattern = np.array(growthPattern)

plt.figure(figsize=(5,5),dpi=100)
for i in range(len(growthPattern)):
    fromD = growthPattern[i][0]
    toD = growthPattern[i][1]
    stacked = np.stack((fromD,toD),axis=1)
    plt.plot(stacked[0],stacked[1],color="black",linewidth=1.1)

ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
ax.set_aspect('equal', adjustable='box')

plt.xlim(0,L)
plt.ylim(0,L)

plt.savefig("dendrite.png")
plt.close()
