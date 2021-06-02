import numpy as np
import subprocess
import multiprocessing as mp


def trentoRun(params, nTrent, uncert=False):
    string = '../build/src/trento Pb Pb ' + str(nTrent) + ' -p ' + str(params[0]) + ' -w ' + str(params[1])
    with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
        data = np.array([l.split() for l in proc.stdout], dtype=float)[:, 4]
    with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
        data2 = np.array([l.split() for l in proc.stdout], dtype=float)[:, 5]
    aveg = np.mean(data)
    aveg2 = np.mean(data2)
    if uncert:
        std1 = np.std(data) / np.sqrt(nTrent)
        std2 = np.std(data2) / np.sqrt(nTrent)
        return np.array([(aveg, aveg2), (std1, std2)])
    return np.array([(aveg, aveg2)])


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
pairList = np.array([(8, 65536), (16, 32768), (32, 16384), (64, 8192), (128, 4096), (256, 2048), (512, 1024),
                     (1024, 512), (2048, 256), (4096, 128)])
# pairList = np.array([(10, 1000)])
paramLabels = np.array(["Reduced thickness", "Nucleon-Width"])
paramMins = np.array([0, 0.5])
paramMaxs = np.array([0.5, 1.2])
obsLabels = np.array([r"$\epsilon$2", r"$\epsilon$3"])
expRelUncert = np.array([0.0, 0.0])
theoRelUncert = np.array([0.0, 0.0])
paramTruths = np.array([0.314, 0.618])
obsTruths = trentoRun(paramTruths, 100000, uncert=True)
print(paramTruths[0], paramTruths[1], obsTruths[0], obsTruths[1])


for aa in range(len(pairList)):
    totDesPts = pairList[aa][0]
    nTrentoRuns = pairList[aa][1]  # Number of times to run Trento
    accessFileName = "./2to19/" + str(totDesPts) + "dp" + str(nTrentoRuns) + "tr"
    dataFileName = "./2to19/" + str(totDesPts) + "dp" + str(nTrentoRuns) + "trData"

    # Storage: data file name, amount of Design Points, [parameter names], [parameter min values],
    #          [parameter max values], [parameter truths], [observable names], [observable truths],
    #          [experimental relative uncertainty], [theoretical relative uncertainty],
    #          number of trento runs per design point
    store1 = np.array([dataFileName, totDesPts, paramLabels, paramMins, paramMaxs, paramTruths,
                       obsLabels, obsTruths, expRelUncert, theoRelUncert, nTrentoRuns], dtype=object)

    np.save(accessFileName, store1)
    print("Saved parameters file, dp: " + str(totDesPts) + ", tr: " + str(nTrentoRuns))

    if getData:
        unit_random_sequence = get_quasirandom_sequence(len(paramLabels), totDesPts)
        design_points = np.zeros(np.shape(unit_random_sequence))
        observables = np.zeros((len(design_points), len(obsTruths)))

        def runItBackTurbo(ii):
            for jj in range(len(paramLabels)):
                design_points[ii][jj] = paramMins[jj] + unit_random_sequence[ii][jj] * (paramMaxs[jj] - paramMins[jj])
            observables[ii] = trentoRun(design_points[ii], nTrentoRuns)

        pool = mp.Pool()
        pool.map(runItBackTurbo, range(len(design_points)))

        store2 = np.array([design_points, observables])
        np.save(dataFileName, store2)
        print("Saved design points and observables, dp: " + str(totDesPts) + ", tr: " + str(nTrentoRuns))
    #    plt.plot(design_points[:, 0], design_points[:, 1], 'b.')
    #    plt.show()
