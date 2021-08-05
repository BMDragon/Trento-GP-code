# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.integrate
import scipy.optimize as opt

# Use Gaussian process from scikit-learn
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels

# suppression warning messages
import warnings

warnings.filterwarnings('ignore')


# Make Changes Here #
pairList = np.array([(8, 8192), (16, 4096), (32, 2048), (64, 1024), (128, 512), (256, 256),
                     (512, 128), (1024, 64), (2048, 32)])
folderName = "./2to16/"
emulatorGraphs = True
posteriorGraphs = True

# DO NOT MAKE CHANGES BELOW #
##################################################


def do_something(bb):
    # Storage: [data file names], amount of Design Points, [parameter names], [parameter min values],
    #          [parameter max values], [parameter truths], [observable names], [observable truths],
    #          number of trento runs per design point
    savedValues = np.load("" + folderName + str(pairList[bb][0]) + "dp"
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

        emulator_design_pts_value = np.array(desPts)  # .tolist()

        emulator_obs_mean_value = np.array(observables[:, nn])  # .tolist()

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
            # 'mean':scipy.interpolate.interp2d(calc_d[obs_name]['x_list'], calc_d[obs_name]['y_list'], np.transpose(
            # calc_d[obs_name]['mean']), kind='linear', copy=True, bounds_error=False, fill_value=None),
            # 'uncert':scipy.interpolate.interp2d(calc_d[obs_name]['x_list'], calc_d[obs_name]['y_list'], np.transpose(
            # calc_d[obs_name]['uncert']), kind='linear', copy=True, bounds_error=False, fill_value=None)
        }

        #####################
        # Plot the emulator #
        #####################
        if emulatorGraphs:
            # Label for the observable
            obs_label = obsNames[nn]

            # observable vs value of one parameter (with the other parameter fixed)
            for pl in range(len(paramTruths)):
                plt.figure(1)
                plt.xscale('linear')
                plt.yscale('linear')
                plt.xlabel(paramNames[pl])
                plt.ylabel(obs_label)

                # Compute the posterior for a range of values of the parameter "x"
                ranges = np.zeros(50).reshape((1, 50))
                for rr in range(0, len(paramMins)):
                    if rr != pl:
                        val = (paramMins[rr] + paramMaxs[rr]) / 2
                        ranges = np.append(ranges, np.linspace(val, val, 50).reshape((1, 50)), axis=0)
                    else:
                        ranges = np.append(ranges, np.linspace(paramMins[rr], paramMaxs[rr], 50).reshape((1, 50)), axis=0)

                param_value_array = np.transpose(ranges[1:, :])

                z_list, z_list_uncert = gaussian_process.predict(param_value_array, return_std=True)

                # Plot design points
                plt.errorbar(desPts[:, pl], np.array(observables[:, nn]),
                             yerr=np.array(truthUncert)[:, nn], fmt='D', color='orange', capsize=4)

                # Plot interpolator
                plt.plot(ranges[pl + 1], z_list, color='blue')
                plt.fill_between(ranges[pl + 1], z_list - z_list_uncert, z_list + z_list_uncert, color='blue', alpha=.4)

                # Plot the truth
                plt.plot(paramTruths[pl], obsTruths[nn], "D", color='black')
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                plt.tight_layout()
    plt.show()
    print(str(pairList[bb]) + " emulators trained")

    ### Compute the Posterior ###
    # We assume uniform priors for this example
    # Here 'x' is the only model parameter

    def prior():
        return 1

    # Under the approximations that we're using, the posterior is
    # exp(-1/2*\sum_{observables, pT}
    # (model(observable,pT)-data(observable,pT))^2/(model_err(observable,pT)^2+exp_err(observable,pT)^2)

    # Here 'x' is the only model parameter

    def likelihood(params):
        res = 0.0
        norm = 1.
        # Sum over observables
        for xx in range(len(obsTruths)):
            # Function that returns the value of an observable
            data_mean2 = obsTruths[xx]
            data_uncert2 = truthUncert[xx]
            tmp_data_mean, tmp_data_uncert = data_mean2, data_uncert2

            emulator = emul_d[obsNames[xx]]['gpr']
            tmp_model_mean, tmp_model_uncert = emulator.predict(np.atleast_2d(np.transpose(params)), return_std=True)

            cov = (tmp_model_uncert * tmp_model_uncert + tmp_data_uncert * tmp_data_uncert)
            res += np.power(tmp_model_mean - tmp_data_mean, 2) / cov
            norm *= 1 / np.sqrt(cov.astype('float'))
        res *= -0.5
        e = 2.71828182845904523536
        return norm * e ** res

    def posterior(*params):
        return prior() * likelihood(np.array([*params]))

    # Compute the posterior for a range of values of the parameter "x"
    div = totDesPoints
    if totDesPoints < 50:
        div = 50

    param_ranges = np.zeros((len(paramMins), div))
    for qq in range(len(paramMins)):
        param_ranges[qq] = np.arange(paramMins[qq], paramMaxs[qq], (paramMaxs[qq] - paramMins[qq])/div)

    paramTruthPost = float(posterior(*paramTruths))

    ranges = np.zeros((len(paramMins), 2))
    for dex in range(len(paramMins)):
        ranges[dex][0] = paramMins[dex]
        ranges[dex][1] = paramMaxs[dex]
    vol1 = scipy.integrate.nquad(posterior, [*ranges], opts={'epsrel': 0.01})[0]
    normish = paramTruthPost / vol1
    print(vol1)

    def minPost(*params):
        return -1*posterior(*params[0])

    maxPostPar = opt.fmin(minPost, paramTruths)
    maxPost = float(posterior(*maxPostPar)) / vol1
    AIC = -2 * np.log(maxPost) + 2*len(paramMins)

    subtitle = "NormP: " + str(normish) + ", AIC: " + str(AIC)

    print(str(pairList[bb]) + " normP(truth): " + str(normish))
    print(str(pairList[bb]) + " AIC: " + str(AIC))

    ###############################
    # Plotting marginal posterior #
    ###############################
    if posteriorGraphs:
        for i in range(len(paramNames)):
            plt.figure()
            plt.xscale('linear')
            plt.yscale('linear')
            plt.xlabel(paramNames[i])
            plt.ylabel(r'Posterior')
            plt.title("Number of design points: " + str(totDesPoints) + ", Number of trento runs: " + str(nTrento))
            plt.figtext(.5, 0.01, subtitle, ha='center')

            # The marginal posterior for a parameter is obtained by integrating over a subset of other model parameters

            # Compute the posterior for a range of values of the parameter "param_1"
            posterior_list = np.array([])
            param_vals = np.array([*paramTruths])
            for ee in param_ranges[i]:
                param_vals[i] = ee
                posterior_list = np.append(posterior_list, posterior(*param_vals))
            plt.plot(param_ranges[i], posterior_list, "-", color='black', lw=4)
            plt.axvline(x=paramTruths[i], color='red')
            plt.tight_layout()
        plt.show()


pool = mp.Pool()
pool.map(do_something, range(len(pairList)))
