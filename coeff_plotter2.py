import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import pickle

# Read in data: DataIN
filename = 'norm_coeffs.csv'
FMatrix = np.genfromtxt(filename, delimiter=',')
FMatrix = FMatrix[1:,1:]
nF = FMatrix.shape[0]
nC = FMatrix.shape[1]

# Read in headers: Features & Cycles
f = open(filename, 'rt')
reader = csv.reader(f)
Cycles = next(reader, None)


MODEL = 'RF'
#models = pickle.load(open(MODEL+"_trained_models.pkl", "rb" ))
#for i in np.arange(nC):
#    FMatrix[:,i] = models[i].feature_importances_
#
#models = pickle.dump(FMatrix, open(MODEL+"_features_coeffs.pkl", "wb" ))

FMatrix = pickle.load(open(MODEL+"_features_coeffs.pkl", "rb" ))



column = {}
for h in Cycles:
    column[h] = []
Features = []
for row in reader:
    for h, v in zip(Cycles, row):
        Features.append(v)
nDel = 0
for i in range(nF):
    for j in range(nC+1):
        if j == 0:
            nDel += 1
        else:
            del Features[nDel]
del Cycles[0]



FMatrix = abs(FMatrix*100)
for i in range(nF):
    for j in range(nC):
        if FMatrix[i,j] == 0:
            FMatrix[i,j] = float('nan')


# Plot
fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(FMatrix)
matplotlib.rcParams.update({'font.size': 18})

# Create colorbar
cbarlabel="Feature weight * 100"
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", size=18)

# We want to show all ticks...
ax.set_xticks(np.arange(len(Cycles)))
ax.set_yticks(np.arange(len(Features)))
# ... and label them with the respective list entries
ax.set_xticklabels(Cycles)
ax.set_yticklabels(Features)

# Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(nF):
    for j in range(nC):
        if FMatrix[i,j] != 0:
            text = ax.text(j, i, np.round(FMatrix[i,j],1), ha="center", va="center", color="w")

ax.set_title("Feature selection for random forest", weight='bold', size=20)
fig.tight_layout()
plt.show()
plt.xlabel('Cycle number')

plt.savefig(MODEL + '_features.png')