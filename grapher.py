import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

pairList = np.array([(8, 8192), (16, 4096), (32, 2048), (64, 1024), (128, 512), (256, 256),
                     (512, 128), (1024, 64), (2048, 32)])
folderList = np.array(["2to16", "3d", "3de4", "3de5", "AuAu", "datUncert", "datUncert2",
                       "e4", "e5", "kw"])
mm = 0  # 0 for mode, 1 for mean


def do_something(fc):
    print("running " + folderList[fc])
    savedValues = np.load("./" + folderList[fc] + "/" + str(pairList[0][0]) + "dp"
                          + str(pairList[0][1]) + "tr.npy", allow_pickle=True)
    paramNames = savedValues[2]
    paramMins = savedValues[3]
    paramMaxs = savedValues[4]
    paramTruths = savedValues[5]
    stats = np.zeros((len(pairList), 5, len(paramMins)))
    for bb in range(len(pairList)):
        file = np.load("./" + folderList[fc] + "/" + str(pairList[bb][0]) + "dp" +
                       str(pairList[bb][1]) + "trStats.npy", allow_pickle=True)
        fil2 = np.load("./" + folderList[fc] + "/" + str(pairList[bb][0]) + "dp" +
                       str(pairList[bb][1]) + "trAICandNorm.npy", allow_pickle=True)
        stats[bb][0] = file[0]
        stats[bb][1] = file[1]
        stats[bb][2] = file[2]
        stats[bb][3][0] = fil2[0]
        stats[bb][3][1] = fil2[1]
        stats[bb][4][0] = np.sqrt(sum([((file[mm][ii] - paramTruths[ii])/paramTruths[ii])**2
                                       for ii in range(len(paramMins))]))
        stats[bb][4][1] = np.sqrt(sum([(np.sqrt(file[2][ii])/paramTruths[ii])**2 for ii in range(len(paramMins))]))

    plt.rc('font', size=16)
    fig, axs = plt.subplots(len(paramMins)+1, sharex='all', constrained_layout=True)
    fig.set_size_inches(5.5, 3.3+1.32*len(paramMins))
    
    
    plt.xlabel("Number of Design Points")
    for i in range(len(paramMins)):
        axs[i].errorbar(pairList[:, 0], stats[:, mm, i], fmt='D', ms=4,
                        yerr=[np.sqrt(np.array(stats[:, 2, i])), np.sqrt(np.array(stats[:, 2, i]))])
        axs[i].axhline(y=paramTruths[i], color='r', linestyle='-', alpha=.5)
        axs[i].set(ylabel=paramNames[i].replace(' ', '\n'))
        axs[i].set_xscale('log', base=2)
        axs[i].set_ylim(paramMins[i], paramMaxs[i])

    hold = len(paramMins)
    axs[hold].plot(pairList[:, 0], stats[:, 3, 1], label='Posterior')
    axs[hold].set(ylabel='Posterior at\nthe truth')
    axs[hold].set_xscale('log', base=2)
    axs[hold].set_ylim(0, max(stats[:, 3, 1])*1.5)
    axs[hold].legend(loc=2, fontsize=13)

    dub = axs[hold].twinx()
    dub.plot(pairList[:, 0], stats[:, 3, 0], 'r', label='AIC')
    dub.set(ylabel='Akaike Information\nCriterion')
    dub.set_ylim(min(stats[:, 3, 0])-0.5, max(stats[:, 3, 0])/3)
    dub.legend(loc=1, fontsize=13)

    axs[1].set_ylim(0.5, 0.8)
    axs[1].set(ylabel='Nucleon width')
    # plt.show()
    savepath = "/mnt/c/Users/bmwei/Pictures/QCD Images/output graphs/" + folderList[fc] + " output.png"
    plt.savefig(savepath, dpi=300)
    print("made " + folderList[fc])


pool = mp.Pool()
pool.map(do_something, range(len(folderList)))
