# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.integrate

# Use Gaussian process from scikit-learn
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels

# suppression warning messages
import warnings

warnings.filterwarnings('ignore')

pairList = np.array([(8, 65536), (16, 32768), (32, 16384), (64, 8192), (128, 4096), (256, 2048), (512, 1024)])
# pairList = np.array([(10, 1000)])


def do_something(bb):
    # Storage: data file name, Total amount of Design Points, [parameter names], [parameter min values],
    #          [parameter max values], [parameter truths], [observable names], [observable truths],
    #          [experimental relative uncertainty], [theoretical relative uncertainty],
    #          number of trento runs per design point
    # savedValues = np.load("listedVerySmall.npy", allow_pickle=True)
    savedValues = np.load("./2to19/" + str(pairList[bb][0]) + "dp" + str(pairList[bb][1]) + "tr.npy", allow_pickle=True)
    totDesPoints = savedValues[1]
    paramNames = savedValues[2]
    paramMins = savedValues[3]
    paramMaxs = savedValues[4]
    paramTruths = savedValues[5]
    obsNames = savedValues[6]
    obsTruths = savedValues[7][0]
    truthUncert = savedValues[7][1]
    expRelUncert = savedValues[8]
    theoRelUncert = savedValues[9]
    nTrento = savedValues[10]

    #   datum: np.array([[design_points], [observables]])
    datum = np.load(str(savedValues[0]) + ".npy", allow_pickle=True)
    desPts = datum[0]
    observables = datum[1]

    #    desPts = np.load(str(savedValues[0][0]) + ".npy", allow_pickle=True)
    #    observables = np.load(str(savedValues[0][1]) + ".npy", allow_pickle=True)

    ### Add uncertainty to the observables ###
    calcUncertList = np.multiply(observables, theoRelUncert)
    noise = np.zeros(np.shape(calcUncertList))
    for ii in range(len(calcUncertList)):
        for jj in range(len(obsTruths)):
            noise[ii][jj] = np.random.normal(0, calcUncertList[ii][jj])
    calcMeanPlusNoise = np.add(observables, noise)

    ### Make emulator for each observable ###
    emul_d = {}

    for nn in range(len(obsTruths)):
        # Label for the observable
        obs_label = obsNames[nn]

        # Function that returns the value of an observable (just to get the truth)

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

        # https://github.com/keweiyao/JETSCAPE2020-TRENTO-BAYES/blob/master/trento-bayes.ipynb
        print('Information on emulator for observable ' + obs_label)
        print('RBF: ', gaussian_process.kernel_.get_params()['k1'])
        print('White: ', gaussian_process.kernel_.get_params()['k2'])

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
                         yerr=np.array(calcUncertList)[:, nn], fmt='D', color='orange', capsize=4)

            # Plot interpolator
            plt.plot(ranges[pl + 1], z_list, color='blue')
            plt.fill_between(ranges[pl + 1], z_list - z_list_uncert, z_list + z_list_uncert, color='blue', alpha=.4)

            # Plot the truth
            plt.plot(paramTruths[pl], obsTruths[nn], "D", color='black')
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.tight_layout()
    plt.close(1)

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

    def posterior(params):
        return prior() * likelihood(params)

    ### Plot the Posterior ###
    # Info about parameters
    param1_label = paramNames[0]
    param1_truth = paramTruths[0]

    param2_label = paramNames[1]
    param2_truth = paramTruths[1]

    plt.figure()
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel(param1_label)
    plt.ylabel(param2_label)
    plt.title("Number of design points: " + str(totDesPoints) + ", Number of trento runs: " + str(nTrento))

    # Compute the posterior for a range of values of the parameter "x"
    div = totDesPoints
    if totDesPoints < 50:
        div = 50

    param1_range = np.arange(paramMins[0], paramMaxs[0], (paramMaxs[0] - paramMins[0]) / div)
    param2_range = np.arange(paramMins[1], paramMaxs[1], (paramMaxs[1] - paramMins[1]) / div)

    param1_mesh, param2_mesh = np.meshgrid(param1_range, param2_range, sparse=False, indexing='ij')

    posterior_array = np.array([posterior((param1_val, param2_val)) for (param1_val, param2_val) in
                                zip(param1_mesh, param2_mesh)])

    paramTruthPost = float(posterior(paramTruths))
    maxPost = np.amax(posterior_array)
    print(str(pairList[bb]) + "Posterior at parameter truth: " + str(paramTruthPost))
    print(str(pairList[bb]) + "Max Posterior: " + str(maxPost))
    ratio = paramTruthPost/maxPost

    def trapezoid(myArr, dx):
        add = np.sum(myArr) - 0.5 * myArr[0] - 0.5 * myArr[-1]
        return add * dx

    temp = np.zeros((len(posterior_array)))
    for zz in range(len(posterior_array)):
        temp[zz] = trapezoid(posterior_array[zz], param2_mesh[0][1] - param2_mesh[0][0])

    vol = trapezoid(temp, param1_mesh[1][0] - param1_mesh[0][0])
    norm = paramTruthPost/vol
    hellDistance = np.sqrt(1 - np.sqrt(paramTruthPost / np.sum(posterior_array)))
    print(str(pairList[bb]) + "Hellinger Distance: " + str(hellDistance))

    print(str(pairList[bb]) + ", normP(truth): " + str(norm))

    # Plot the posterior
    cs = plt.contourf(param1_mesh, param2_mesh, posterior_array, levels=20)
    cbar = plt.colorbar(cs, label="Posterior")
    plt.plot([param1_truth], [param2_truth], "D", color='red', ms=10)
    # plt.figtext(.5, 0.01, subtitle, ha='center')
    plt.tight_layout()
    plt.close()

    """
    ###############################
    # Plotting marginal posterior #
    ###############################
    for i in range(len(paramNames)):
        plt.figure()
        plt.xscale('linear')
        plt.yscale('linear')
        plt.xlabel(paramNames[i])
        plt.ylabel(r'Posterior')
        plt.title("Number of design points: " + str(totDesPoints) + ", Number of trento runs: " + str(nTrento))

        # The marginal posterior for a parameter is obtained by integrating over a subset of other model parameters

        # Compute the posterior for a range of values of the parameter "param_1"
        param_range = np.linspace(paramMins[i], paramMaxs[i], div)
        posterior_list = np.array([])

        if i == 0:
            posterior_list = np.array([scipy.integrate.quad(lambda param2_val: posterior((param1_val, param2_val)),
                                                            paramMins[1], paramMaxs[1])[0] for param1_val in param_range])
        elif i == 1:
            posterior_list = np.array([scipy.integrate.quad(lambda param1_val: posterior((param1_val, param2_val)),
                                                            paramMins[0], paramMaxs[0])[0] for param2_val in param_range])

        plt.plot(param_range, posterior_list, "-", color='black', lw=4)
        plt.axvline(x=paramTruths[i], color='red')
        plt.tight_layout()
    plt.close()
    """


pool = mp.Pool()
pool.map(do_something, range(len(pairList)))
