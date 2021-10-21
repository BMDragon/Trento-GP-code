# Load libraries
import numpy as np
import multiprocessing as mp

import scipy.integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Use Gaussian process from scikit-learn
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels

# suppression warning messages
import warnings

warnings.filterwarnings('ignore')


# Make Changes Here #
pairList = np.array([(512, 128)])
"""folderList = np.array(["2to16", "3d", "3de4", "3de5", "4d", "AuAu", "datUncert", "datUncert2",
                       "e4", "e5"])"""

folderList = np.array(["4d"])

emulatorGraphs = True
posteriorGraphs = True


def do_something(bb):
    # Storage: [data file names], amount of Design Points, [parameter names], [parameter min values],
    #          [parameter max values], [parameter truths], [observable names], [observable truths],
    #          number of trento runs per design point
    for fc in range(len(folderList)):
        savedValues = np.load("./" + folderList[fc] + "/" + str(pairList[bb][0]) + "dp"
                              + str(pairList[bb][1]) + "tr.npy", allow_pickle=True)
        totDesPoints = savedValues[1]
        paramNames = savedValues[2]
        paramMins = savedValues[3]
        paramMaxs = savedValues[4]
        paramTruths = savedValues[5]
        obsNames = savedValues[6]
        obsTruths = savedValues[7][0]
        truthUncert = savedValues[7][1]
        nTrento = savedValues[8]

        #   datum: np.array([[design_points], [observables]])
        desPts = np.load(str(savedValues[0][0]) + ".npy", allow_pickle=True)
        observables = np.load(str(savedValues[0][1]) + ".npy", allow_pickle=True)

        ### Make emulator for each observable ###
        emul_d = {}
        for nn in range(len(obsTruths)):
            # Kernels
            k0 = 1. * kernels.RBF(
                # length_scale=(param1_paramspace_length / 2., param2_paramspace_length / 2.)
                #    length_scale_bounds=(
                #        (param1_paramspace_length / param1_nb_design_pts, 3. * param1_paramspace_length),
                #        (param2_paramspace_length / param2_nb_design_pts, 3. * param2_paramspace_length)
                #    )
            )

            k2 = 1. * kernels.WhiteKernel(
                noise_level=truthUncert[nn],
                # noise_level_bounds='fixed'
                noise_level_bounds=(truthUncert[nn] / 4., 4 * truthUncert[nn])
            )

            kernel = (k0 + k2)
            nrestarts = 10
            emulator_design_pts_value = np.array(desPts)
            emulator_obs_mean_value = np.array(observables[:, nn])

            # Fit a GP (optimize the kernel hyperparameters) to each PC.
            gaussian_process = GPR(
                kernel=kernel,
                alpha=0.0001,
                n_restarts_optimizer=nrestarts,
                copy_X_train=True
            ).fit(emulator_design_pts_value, emulator_obs_mean_value)
            """
            # https://github.com/keweiyao/JETSCAPE2020-TRENTO-BAYES/blob/master/trento-bayes.ipynb
            print('Information on emulator for observable ' + obs_label)
            print('RBF: ', gaussian_process.kernel_.get_params()['k1'])
            print('White: ', gaussian_process.kernel_.get_params()['k2'])
            """

            emul_d[obsNames[nn]] = {
                'gpr': gaussian_process
                # 'mean':scipy.interpolate.interp2d(calc_d[obs_name]['x_list'], calc_d[obs_name]['y_list'],
                # np.transpose(calc_d[obs_name]['mean']), kind='linear', copy=True, bounds_error=False,
                # fill_value=None), 'uncert':scipy.interpolate.interp2d(calc_d[obs_name]['x_list'],
                # calc_d[obs_name]['y_list'], np.transpose(calc_d[obs_name]['uncert']), kind='linear',
                # copy=True, bounds_error=False, fill_value=None)
            }
        print(str(pairList[bb]) + " emulators trained")

        ### Compute the Posterior ###
        # We assume uniform priors (integral across the whole parameter space should = 1)

        dims = np.array([paramMaxs[vv] - paramMins[vv] for vv in range(len(paramMins))])
        area = np.prod(dims)
        height = 1.0/area

        def prior():
            return height

        # Under the approximations that we're using, the posterior is
        # Likelihood = exp((-1/2)ln((2 pi)^n\prod_n{modelErr(observable, pT)^2 + dataErr(observable, pT)^2})
        #                 - (1/2)\sum_{observables, pT}(model(observable, pT) - data(observable, pT))^2
        #                 / (modelErr(observable, pT)^2 + dataErr(observable, pT)^2))

        def likelihood(params):
            res = 0.0
            norm = (2*np.pi)**len(obsTruths)

            # Sum over observables
            for xx in range(len(obsTruths)):
                # Function that returns the value of an observable
                data_mean2 = obsTruths[xx]
                data_uncert2 = truthUncert[xx]
                tmp_data_mean, tmp_data_uncert = data_mean2, data_uncert2

                emulator = emul_d[obsNames[xx]]['gpr']
                tmp_model_mean, tmp_model_uncert = emulator.predict(np.atleast_2d(np.transpose(params)),
                                                                    return_std=True)

                cov = (tmp_model_uncert * tmp_model_uncert + tmp_data_uncert * tmp_data_uncert)
                res += np.power(tmp_model_mean - tmp_data_mean, 2) / cov
                norm *= cov
            res *= -0.5

            return (norm ** -0.5) * (np.e ** res)

        def posterior(*params):
            return prior() * likelihood(np.array([*params]))

        def minPost(*params):
            return -1*posterior(*params[0])

        maxPostPar = opt.fmin(minPost, paramTruths)
        print(str(pairList[bb]) + " Mode: " + str(maxPostPar))
        div = totDesPoints
        if totDesPoints < 50:
            div = 50
        param_ranges = np.zeros((len(paramMins), div))
        for qq in range(len(paramMins)):
            param_ranges[qq] = np.arange(paramMins[qq], paramMaxs[qq], (paramMaxs[qq] - paramMins[qq]) / div)
        num = 0

        def meanParam(*params):
            return (params[num])*posterior(*params)

        ranges = np.zeros((len(paramMins), 2))
        for dex in range(len(paramMins)):
            ranges[dex][0] = paramMins[dex]
            ranges[dex][1] = paramMaxs[dex]

        mean = np.zeros((len(paramMins)))
        vol1 = scipy.integrate.nquad(posterior, [*ranges], opts={'epsrel': 0.01})[0]
        for pp in range(len(paramMins)):
            num = pp
            mean[pp] = scipy.integrate.nquad(meanParam, [*ranges], opts={'epsrel': 0.01})[0]/vol1

        print(str(pairList[bb]) + " Mean: " + str(mean))

        def varParam(*params):
            return posterior(*params)*(params[num] - mean[num])**2

        var = np.zeros((len(paramMins)))
        for pp in range(len(paramMins)):
            num = pp
            var[pp] = scipy.integrate.nquad(varParam, [*ranges], opts={'epsrel': 0.01})[0]/vol1

        print(str(pairList[bb]) + " Variance: " + str(var))

        maxLike = float(likelihood(maxPostPar))
        AIC = -2 * np.log(maxLike) + 2 * len(paramMins)
        paramTruthPost = float(posterior(*paramTruths))
        normish = paramTruthPost / vol1

        print(str(pairList[bb]) + " normP(truth): " + str(normish))
        print(str(pairList[bb]) + " AIC: " + str(AIC))
        AICandNorm = np.array([AIC, normish])

        store = np.array([maxPostPar, mean, var], dtype=object)
        saveFileName = "./" + folderList[fc] + "/" + str(pairList[bb][0]) + "dp" + \
                       str(pairList[bb][1]) + "trStats"
        saveFileName2 = "./" + folderList[fc] + "/" + str(pairList[bb][0]) + "dp" + \
                        str(pairList[bb][1]) + "trAICandNorm"
        np.save(saveFileName, store)
        np.save(saveFileName2, AICandNorm)

    print(str(pairList[bb]) + " Done")


# Use multiprocessing to make the script run faster
pool = mp.Pool()
pool.map(do_something, range(len(pairList)))
