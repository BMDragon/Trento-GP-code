import numpy as np
import subprocess
import matplotlib.pyplot as plt


def trentoRun(params):
    string = '../build/src/trento Pb Pb ' + str(nTrentoRuns) + ' -p ' + str(params[0]) + ' -w ' + str(params[1])
    with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
        data = np.array([l.split() for l in proc.stdout], dtype=float)[:, 4]
    with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
        data2 = np.array([l.split() for l in proc.stdout], dtype=float)[:, 5]
    aveg = np.mean(data)
    aveg2 = np.mean(data2)
    return np.array([aveg, aveg2])


def chm(params):
    aveg = params[0]
    aveg2 = params[1]
    return np.array([aveg, aveg2])


def get_quasirandom_sequence(dim, num_samples):
    def phi(dd):
        x = 2.0000
        for iii in range(10):
            x = pow(1 + x, 1 / (dd + 1))
            return x

    d = dim  # Number of dimensions
    n = num_samples  # Array of number of design points for each parameter

    g = phi(d)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = pow(1 / g, j + 1) % 1

    z = np.zeros((n, d))

    # This number can be any real number.
    # Common default setting is typically seed=0
    # But seed = 0.5 is generally better.
    seed = 0.5
    for i in range(len(z)):
        z[i] = (seed + alpha * (i + 1)) % 1

    return z


getData = True
accessFileName = "listedTrento"
dataFileName = "listDataTrento"
paramLabels = np.array(["Reduced thickness", "Nucleon-Width"])
totDesPts = 250
nTrentoRuns = 4000  # Number of times to run Trento
paramMins = np.array([0, 0.5])
paramMaxs = np.array([0.5, 1.2])
paramTruths = np.array([0.314, 0.618])
obsLabels = np.array([r"$\epsilon$2", r"$\epsilon$3"])
expRelUncert = np.array([0.05, 0.05])
theoRelUncert = np.array([0.05, 0.05])

obsTruths = trentoRun(paramTruths)
print(paramTruths[0], paramTruths[1], obsTruths[0], obsTruths[1])

# Storage: data file name, amount of Design Points, [parameter names], [parameter min values],
#          [parameter max values], [parameter truths], [observable names], [observable truths],
#          [experimental relative uncertainty], [theoretical relative uncertainty]
store1 = np.array([dataFileName, totDesPts, paramLabels, paramMins, paramMaxs, paramTruths,
                   obsLabels, obsTruths, expRelUncert, theoRelUncert], dtype=object)

np.save(accessFileName, store1)
print("Saved parameters file")

if getData:
    unit_random_sequence = get_quasirandom_sequence(len(paramLabels), totDesPts)
    design_points = np.zeros(np.shape(unit_random_sequence))
    observables = np.zeros((len(design_points), len(obsTruths)))
    for ii in range(len(design_points)):
        for jj in range(len(paramLabels)):
            design_points[ii][jj] = paramMins[jj] + unit_random_sequence[ii][jj] * (paramMaxs[jj] - paramMins[jj])
        observables[ii] = trentoRun(design_points[ii])
    store2 = np.array([design_points, observables], dtype=object)
    np.save(dataFileName, store2)
    print("Saved design points and observables")
    plt.plot(design_points[:, 0], design_points[:, 1], 'b.')
    plt.show()
