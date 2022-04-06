import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger
from keras.models import Model
import matplotlib.patches as mpatches
import sys
from tqdm import tqdm
from matplotlib.colors import SymLogNorm, LogNorm

np.set_printoptions(threshold=sys.maxsize)

def cstm_PuBu(x):
    return plt.cm.PuBu((np.clip(x,2,10)-2)/8.)

def cstm_RdBu(x):
    return plt.cm.RdBu((np.clip(x,2,10)-2)/8.)

def PlotLoss(epochs, loss_training, loss_validation, path):
    plt.figure()
    plt.grid(alpha = 0.2)
    plt.plot(epochs, loss_training, label = "Training")
    plt.plot(epochs, loss_validation, label = "Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotEnergyScattering2D(energy_true, energy_rec, path):
    plt.figure()
    plt.grid(alpha = 0.2)
    x = np.linspace(np.min(energy_true), np.max(energy_true), 100)
    plt.plot(x, x, color="black")
    # plt.scatter(x,y,edgecolors='none',s=marker_size,c=void_fraction, norm=matplotlib.colors.LogNorm())
    plt.hist2d(energy_true, energy_rec, bins=(50, 50), cmap = "viridis", norm = matplotlib.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Number of events')
    plt.xlabel("$\log_{10}(E_\mathrm{true}/\mathrm{GeV})$")
    plt.ylabel("$\log_{10}(E_\mathrm{rec}/\mathrm{GeV})$")
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotRelativeEnergyError(relative_energy_error_single, mean, sigma_total, path):
    plt.figure()
    plt.grid(alpha = 0.2)
    plt.hist(relative_energy_error_single, bins = np.linspace(np.min(relative_energy_error_single), np.max(relative_energy_error_single), 40))
    # plt.yscale("log")
    plt.xlabel("($E_\mathrm{rec} - E_\mathrm{true})/ E_\mathrm{true}$")
    plt.ylabel("Number of events")
    plt.text(0.95, 0.95, "median $ = %.3f$" % mean, ha = "right", va = "top", transform = plt.gca().transAxes)
    plt.text(0.95, 0.90, "$\sigma = %.3f$" % sigma_total, ha = "right", va = "top", transform = plt.gca().transAxes)
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotRelativeEnergyErrorBinned(energy_true_binned, energy_rec_binned, bins, path):
    fig, ax = plt.subplots(3, 3)
    ax = ax.ravel()
    for j in range(len(energy_true_binned)):
        relative_energy_error = (energy_rec_binned[j] - energy_true_binned[j]) / energy_true_binned[j] 
        median = np.median(relative_energy_error)
        sigma = np.std(relative_energy_error)
        ax[j].set_title(f"{np.round(bins[j], 2)} TeV < $E_{{true}}$ < {np.round(bins[j+1], 2)} TeV", fontsize = 6)
        ax[j].grid(alpha = 0.2)
        ax[j].hist(relative_energy_error, bins = np.linspace(np.min(relative_energy_error), np.max(relative_energy_error), 40))
        ymin, ymax = ax[j].get_ylim()
        ax[j].vlines(median, ymin, ymax, color = "black", linestyle = '-', linewidth = 0.7,  label = "median$ = %.3f$" % median)
        ax[j].vlines(median - sigma, ymin, ymax, color = "black", linestyle = '--', linewidth = 0.7, label = r"$\sigma = %.3f$" % sigma)
        ax[j].vlines(median + sigma, ymin, ymax, color = "black", linestyle = '--', linewidth = 0.7)
        ax[j].tick_params(axis='both', which='major', labelsize = 6)
        ax[j].legend(fontsize = 6)
        ymax = 1.6 * ymax
        ax[j].set_ylim(ymin, ymax)
    ax[-2].set_xlabel("$(E_\mathrm{rec} - E_\mathrm{true}) / E_\mathrm{true}$", fontsize = 6)
    ax[3].set_ylabel("Number of events", fontsize = 6)
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def MedianSigma68(energy_true_binned, energy_rec_binned, bins):
    median_collection = np.array([])
    sigma_collection = np.array([])
    for j in range(len(energy_true_binned)):
        relative_energy_error = (energy_rec_binned[j] - energy_true_binned[j]) / energy_true_binned[j] 
        median = np.median(relative_energy_error)
        median_collection = np.append(median_collection, median)
        relative_energy_error_corrected = np.abs((energy_rec_binned[j] - energy_true_binned[j] - median) / energy_true_binned[j])
        relative_energy_error_corrected = np.sort(relative_energy_error_corrected)
        index_68 = int(len(relative_energy_error_corrected) * 0.68)
        sigma_single = relative_energy_error_corrected[index_68]
        sigma_collection = np.append(sigma_collection, sigma_single)

    return (median_collection, sigma_collection)

  
def PlotRelativeEnergyErrorBinnedCorrected(energy_true_binned, energy_rec_binned, bins, path):
    fig, ax = plt.subplots(3, 3)
    ax = ax.ravel()
    sigma_collection = np.array([])
    for j in range(len(energy_true_binned)):
        relative_energy_error = (energy_rec_binned[j] - energy_true_binned[j]) / energy_true_binned[j] 
        median = np.median(relative_energy_error)
        relative_energy_error_corrected = np.abs((energy_rec_binned[j] - energy_true_binned[j] - median) / energy_true_binned[j])
        relative_energy_error_corrected = np.sort(relative_energy_error_corrected)
        index_68 = int(len(relative_energy_error_corrected) * 0.68)

        sigma_single = relative_energy_error_corrected[index_68]
        sigma_collection = np.append(sigma_collection, sigma_single)
        
        ax[j].set_title(f"{np.round(bins[j], 2)} TeV < E < {np.round(bins[j+1], 2)} TeV", fontsize = 6)
        ax[j].grid(alpha = 0.2)
        ax[j].hist(relative_energy_error_corrected, bins = np.linspace(np.min(relative_energy_error_corrected), np.max(relative_energy_error_corrected), 40))
        ymin, ymax = ax[j].get_ylim()
        ax[j].vlines(sigma_single, ymin, ymax, color = "black", linestyle = '--', linewidth = 0.7,  label = "$\sigma_{68} = %.3f$" % sigma_single)
        ax[j].tick_params(axis='both', which='major', labelsize = 6)
        ax[j].legend(fontsize = 6)
        ymax = 1.3 * ymax
        ax[j].set_ylim(ymin, ymax)
    ax[-2].set_xlabel("$(E_\mathrm{rec} - E_\mathrm{true}) / E_\mathrm{true}$", fontsize = 6)
    ax[3].set_ylabel("Number of events", fontsize = 6)
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()


def PlotEnergyAccuracy(median, bins, path):
    bins_central = np.array([])
    for b in range(len(bins) - 1):
        bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    plt.figure()
    plt.grid(alpha = 0.2)
    plt.errorbar(bins_central, median, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".")
    plt.xlabel("Energy [TeV]")
    plt.ylabel("median$(\Delta E / E_\mathrm{true})$")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()


def PlotEnergyResolution(sigma, bins, path):
    bins_central = np.array([])
    for b in range(len(bins) - 1):
        bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    plt.figure()
    plt.grid(alpha = 0.2)
    plt.errorbar(bins_central, sigma, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".")
    plt.xlabel("Energy [TeV]")
    plt.ylabel("$(\Delta E / E_\mathrm{true})_{68}$")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()


def PlotEnergyAccuracyComparison(median_all, bins, label, path):
    bins_central = np.array([])
    for b in range(len(bins) - 1):
        bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    plt.figure()
    plt.grid(alpha = 0.2)
    for i in range(len(median_all)):
        plt.errorbar(bins_central, median_all[i], xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".", label = label[i])
    plt.xlabel("Energy [TeV]")
    plt.ylabel("median$(\Delta E / E_\mathrm{true})$")
    plt.xscale("log")
    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim(ymin, 1.2 * ymax)
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()


def PlotEnergyResolutionComparison(sigma_all, bins, label, path):
    # mean = np.array([])
    # error = np.array([])
    # for k in range(len(sigma_all[0])):
    #     row = [row_k[k] for row_k in sigma_all]
    #     mean = np.append(mean, nan(row))
    #     error = np.append(error, np.std(row))
    bins_central = np.array([])
    for b in range(len(bins) - 1):
        bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    plt.figure()
    plt.grid(alpha = 0.2)
    for i in range(len(sigma_all)):
        plt.errorbar(bins_central, sigma_all[i], xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".", label = label[i])
    # plt.plot(bins_central, mean, linestyle = "--", label = "Mean", color = "black")
    # plt.fill_between(bins_central, mean - error, mean + error, color = "grey", alpha = 0.25)
    plt.xlabel("Energy [TeV]")
    plt.ylabel("$(\Delta E / E_\mathrm{true})_{68}$")
    plt.xscale("log")
    # plt.ylim(0.2, 0.60)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotEnergyResolutionMean(args_input, sigma_all, bins, label, path):
    table = []
    for k in range(len(args_input)):
        table.append([args_input[k], sigma_all[k]])

    table = pd.DataFrame(table, columns=["input", "energy resolution"])

    table_mean = []
    args_input_unique = np.unique(args_input)
    for k in range(len(args_input_unique)):
        table_k = table.copy()
        table_k.where(table_k["input"] == args_input_unique[k], inplace = True)
        # print(table_k["energy resolution"].to_numpy())
        # print(np.mean(table_k["energy resolution"].to_numpy(), axis = 0))
        # print(np.nanmean(table_k["energy resolution"].to_numpy(), axis = 0))
        # print(np.std(table_k["energy resolution"].to_numpy(), axis = 0))
        # print(np.nanstd(table_k["energy resolution"].to_numpy(), axis = 0))
        table_mean.append([args_input_unique[k], np.mean(table_k["energy resolution"].dropna().to_numpy(), axis = 0), np.std(table_k["energy resolution"].dropna().to_numpy(), axis = 0)])
    table_mean = pd.DataFrame(table_mean, columns=["input", "mean energy resolution", "std energy resolution"])


    bins_central = np.array([])
    for b in range(len(bins) - 1):
        bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    plt.figure()
    plt.grid(alpha = 0.2)
    for i in range(len(args_input_unique)):
        table_mean_i = table_mean.copy()
        mean_energy_resolution = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean energy resolution"].dropna().to_numpy()[0]
        std_energy_resolution = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std energy resolution"].dropna().to_numpy()[0]
        plt.errorbar(bins_central, mean_energy_resolution, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".", label = args_input_unique[i])
        plt.fill_between(bins_central, mean_energy_resolution - std_energy_resolution, mean_energy_resolution + std_energy_resolution, color = "grey", alpha = 0.25)
    plt.xlabel("Energy [TeV]")
    plt.ylabel("$(\Delta E / E_\mathrm{true})_{68}$")
    plt.xscale("log")
    # plt.ylim(0.2, 0.60)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def LoadExampleData(run, string_input, particle_type, string_ps_input, string_input_short, string_data_type, string_table_column):
    # load example data
    run_filename = f"{particle_type}_20deg_0deg_run{run}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
    input_filename = f"dm-finder/cnn/{string_input}/input/{particle_type}/" + string_ps_input + run_filename + string_input_short + string_data_type + ".h5"

    table = pd.read_hdf(input_filename, ignore_index = True)

    # input features
    X = [[]] * len(table)
    for j in range(len(table)):
        X[j] = table[string_table_column][j]
    X = np.asarray(X) 

    # reshape X data
    X_shape = np.shape(X)
    X = X.reshape(-1, X_shape[1], X_shape[2], 1)

    # output label: log10(true energy)
    Y = np.asarray(table["true_energy"])
    Y = np.log10(np.asarray(table["true_energy"]))

    return X, Y

def PlotFeatureMaps(X, model, index_example, path):
    # redefine model to output right after the first hidden layer
    layer_names = [layer.name for layer in model.layers]
    layer_shape = [layer.output_shape for layer in model.layers]
    
    ixs = np.array([], dtype = int)
    for j in range(len(layer_names)):
        if "conv" in layer_names[j]:
            ixs = np.append(ixs, j)

    layer_shape = [layer_shape[j] for j in ixs]
    layer_shape_output = [item[-1] for item in layer_shape]

    outputs = [model.layers[j].output for j in ixs]
    model = Model(inputs = model.inputs, outputs = outputs)

    # load the image with the required shape
    # index_max_energy = np.argmax(Y)
    img = X[index_example]

    plt.figure()
    plt.title("example input")
    plt.xticks([])
    plt.yticks([])
    # plot filter channel in grayscale
    plt.imshow(img, cmap='gray')
    plt.savefig(path + "_0.png", dpi = 250)
    plt.close()

    img = X[index_example].reshape(1, np.shape(X[index_example])[0], np.shape(X[index_example])[1], np.shape(X[index_example])[2])

    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    # plot the output from each block
    for j in range(len(feature_maps)):
        ix = 1
        fig, ax = plt.subplots(int(np.round(np.sqrt(layer_shape_output[j]), decimals = 0)), int(np.round(np.sqrt(layer_shape_output[j]), decimals = 0)))
        fig.suptitle(f"feature maps - {j + 1}. convolutional layer")
        ax = ax.ravel()
        for k in range(layer_shape_output[j]):
            ax[k].set_xticks([])
            ax[k].set_yticks([])
            ax[k].imshow(feature_maps[j][0, :, :, ix-1], cmap='gray')
            ix += 1
        # save the figure
        plt.savefig(path + f"_{j+1}.png", dpi = 250)
        plt.close()

def PlotFilters(model, path):
    #Iterate thru all the layers of the model
    j = 1
    for layer in model.layers:
        if 'conv' in layer.name:
            weights, bias = layer.get_weights()
            
            # normalize filter values between  0 and 1 for visualization
            # f_min, f_max = weights.min(), weights.max()
            # filters = (weights - f_min) / (f_max - f_min)  
            filters = weights[:,:,:, 0]

            fig, ax = plt.subplots(int(np.round(np.sqrt(weights.shape[3]), decimals = 0)), int(np.round(np.sqrt(weights.shape[3]), decimals = 0)))
            fig.suptitle(f"filters - {j}. convolutional layer")
            ax = ax.ravel()
            for k in range(weights.shape[3]):
                filters = weights[:,:,:, k]
                ax[k].set_xticks([])
                ax[k].set_yticks([])
                ax[k].imshow(filters[ :, :, 0], cmap='gray')
            # save the figure
            plt.savefig(path + f"_{j}.png", dpi = 250)
            plt.close()
            j += 1

def EnergyDistributionSeparation(table, path):
    # display total energy distribution of data set
    plt.figure()
    table_gamma = np.asarray(table[table["particle"] == 1].reset_index(drop = True)["true_energy"])
    table_proton = np.asarray(table[table["particle"] == 0].reset_index(drop = True)["true_energy"])
    plt.hist(table_gamma, bins = np.logspace(np.log10(np.min(table_gamma)), np.log10(np.max(table_gamma)), 50), alpha = 0.5, label = "gamma")
    plt.hist(table_proton, bins = np.logspace(np.log10(np.min(table_proton)), np.log10(np.max(table_proton)), 50), alpha = 0.5, label = "proton")
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Number of events")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(path, dpi = 250)
    plt.close()

def EnergyDistributionEnergy(Y, path):
    # display total energy distribution of data set
    plt.figure()
    plt.hist(10**Y, bins=np.logspace(np.log10(np.min(10**Y)),np.log10(np.max(10**Y)), 50))
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Number of events")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(path, dpi = 250)
    plt.close()

def ExamplesEnergy(X, Y, path):
    # plot a few examples
    fig, ax = plt.subplots(3, 3)
    ax = ax.ravel()
    for i in range(9):
        ax[i].imshow(X[i], cmap = "Greys_r")
        ax[i].title.set_text(f"{int(np.round(10**Y[i]))} GeV")
        ax[i].axis("off")
    plt.savefig(path, dpi = 500)

def ExamplesSeparation(X, Y, path):
    # plot a few examples
    fig, ax = plt.subplots(3, 3)
    ax = ax.ravel()
    for i in range(9):
        ax[i].imshow(X[i], cmap = "Greys_r")
        if Y[i][1] == 1:
            ax[i].title.set_text(f"gamma ray")
        elif Y[i][1] == 0:
            ax[i].title.set_text(f"proton")
        ax[i].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi = 500)

def PlotGammaness(gammaness_true, gammaness_rec, path):
    # define true gammaness as boolean
    gammaness_true_bool = gammaness_true.astype(bool)
    gammaness_true_bool_inverted = [not elem for elem in gammaness_true_bool]
    # extract gammaness of true gamma-ray events
    gammaness_gammas = gammaness_rec[gammaness_true_bool]
    # extract gammaness of true proton events
    gammaness_protons = gammaness_rec[gammaness_true_bool_inverted]

    fig, ax = plt.subplots(1)
    plt.grid(alpha = 0.3)
    plt.hist(gammaness_gammas, label = "true photons", bins = np.linspace(0, 1, 31), alpha = 0.6)
    plt.hist(gammaness_protons, label = "true protons", bins = np.linspace(0, 1, 31), alpha = 0.6)
    # plt.axvline(0.5, color = "r", linestyle = "--", label = "decision boundary")
    plt.xlabel("Gammaness")
    plt.ylabel("Number events")
    # plt.yscale("log")
    # handles, labels = fig.gca().get_legend_handles_labels()
    # patch = mpatches.Patch(color = 'white', label = r"total - acc = {0} $\%$".format(np.round(true_positive_rate_total * 100, 2)))
    # handles.append(patch) 
    # plt.legend(handles = handles, framealpha = 0.95)
    plt.legend(framealpha = 0.95)
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PositiveRates(x, y, thresholds):
    true_positive_rate = np.array([])
    false_positive_rate = np.array([])

    for i in range(len(thresholds)):
        true_positive_rate = np.append(true_positive_rate, len(x[x >= thresholds[i]]) / len(x))
        false_positive_rate = np.append(false_positive_rate, len(y[y >= thresholds[i]]) / len(y))

    return(true_positive_rate, false_positive_rate)

def NegativeRates(x, y, thresholds):
    true_negative_rate = np.array([])
    false_negative_rate = np.array([])

    for i in range(len(thresholds)):
        true_negative_rate = np.append(true_negative_rate, len(y[y < thresholds[i]]) / len(y))
        false_negative_rate = np.append(false_negative_rate, len(x[x < thresholds[i]]) / len(x))

    return(true_negative_rate, false_negative_rate)

# area under ROC curve (Riemann integral)
def AreaUnderROCCurve(x, y):
    x = x[::-1]
    y = y[::-1]
    area_under_ROC_curve = np.sum(y[1:] * np.diff(x))
    return area_under_ROC_curve

# def AreaUnderROCCurveNumpy(x, y):
#     area = np.trapz(y[::-1], x[::-1])
#     return area

def PlotGammanessEnergyBinned(table_output, energy_range, path):
    # prepare energy binning
    table_output = table_output.sort_values(by = ["E_true / GeV"], ignore_index = True)
    energy_true = np.asarray(table_output["E_true / GeV"]) * 1e-3
    gammaness_true = np.asarray(table_output["true gammaness"])
    gammaness_rec = np.asarray(table_output["reconstructed gammaness"])

    number_energy_ranges = 6 # number of energy ranges the whole energy range will be splitted
    sst_energy_min = energy_range[0] # TeV
    sst_energy_max = energy_range[1] # TeV
    bins = np.logspace(np.log10(np.min(sst_energy_min)), np.log10(np.max(sst_energy_max)), number_energy_ranges + 1) 
    indices = np.array([], dtype = int)
    for b in range(len(bins) - 2):
        index = np.max(np.where(energy_true < bins[b+1])) + 1
        indices = np.append(indices, index)

    energy_true_binned = np.split(energy_true, indices)
    gammaness_true_binned = np.split(gammaness_true, indices)
    gammaness_rec_binned = np.split(gammaness_rec, indices)

    fig, ax = plt.subplots(3, 2)
    ax = ax.ravel()
    plt.grid(alpha = 0.3)
    for i in range(number_energy_ranges):
        # define true gammaness as boolean
        gammaness_true_bool = gammaness_true_binned[i].astype(bool)
        gammaness_true_bool_inverted = [not elem for elem in gammaness_true_bool]

        # extract gammaness of true gamma-ray events
        gammaness_gammas = gammaness_rec_binned[i][gammaness_true_bool]
        # extract gammaness of true proton events
        gammaness_protons = gammaness_rec_binned[i][gammaness_true_bool_inverted]

        # calculate the true positive rate for gamma rays and protons depending on the threshold
        thresholds = np.linspace(0, 1, 10000)
        true_positive_rate, false_positive_rate = PositiveRates(gammaness_gammas, gammaness_protons, thresholds)
        area_under_ROC_curve = AreaUnderROCCurve(false_positive_rate, true_positive_rate)

        ax[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : 10})
        ax[i].hist(gammaness_gammas, label = "true photons", alpha = 0.6)
        ax[i].hist(gammaness_protons, label = "true protons", alpha = 0.6)
        ylim = ax[i].get_ylim()
        ax[i].set_xlim(-0.05, 1.05)
        ax[i].text(0, 0.8 * ylim[1], f"AUC = {np.round(area_under_ROC_curve, 3)}", fontsize = 6)
        ax[i].axvline(0.5, color = "r", linestyle = "--", label = "decision boundary")
        ax[i].tick_params(axis = 'both', which = 'major', labelsize = 10)

    ax[-1].set_xlabel("Gammaness", fontsize = 10)
    ax[-2].set_xlabel("Gammaness", fontsize = 10)
    ax[2].set_ylabel("Number events", fontsize = 10)
    # plt.legend(framealpha = 0.95, fontsize = 10)
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotROC(gammaness_true, gammaness_rec, path):
    # define true gammaness as boolean
    gammaness_true_bool = gammaness_true.astype(bool)
    gammaness_true_bool_inverted = [not elem for elem in gammaness_true_bool]
    # extract gammaness of true gamma-ray events
    gammaness_gammas = gammaness_rec[gammaness_true_bool]
    # extract gammaness of true proton events
    gammaness_protons = gammaness_rec[gammaness_true_bool_inverted]

    # calculate the true positive rate for gamma rays and protons depending on the threshold
    thresholds = np.linspace(0, 1.0, 10000)
    true_positive_rate, false_positive_rate = PositiveRates(gammaness_gammas, gammaness_protons, thresholds)
    true_negative_rate, false_negative_rate = NegativeRates(gammaness_gammas, gammaness_protons, thresholds)
    true_positive_rate_50, false_positive_rate_50 = PositiveRates(gammaness_gammas, gammaness_protons, np.array([0.5]))
    true_negative_rate_50, false_negative_rate_50 = NegativeRates(gammaness_gammas, gammaness_protons, np.array([0.5]))
    print("Correctly classified photons: ", true_positive_rate_50[0])
    print("Correctly classified protons: ", true_negative_rate_50[0])
    area_under_ROC_curve = AreaUnderROCCurve(false_positive_rate, true_positive_rate)

    # plot the ROC curve
    plt.figure()
    plt.grid(alpha = 0.3)
    plt.plot(false_positive_rate, true_positive_rate, label = "AUC = {0:.3f}".format(np.round(area_under_ROC_curve, 3))) # \nCC $\gamma$ = {1:.3f}\nCC $p$ = {2:.3f}.format(np.round(area_under_ROC_curve, 3), np.round(true_positive_rate_50[0], 3), np.round(true_negative_rate_50[0], 3)))
    plt.plot(np.linspace(0, 1, 5), np.linspace(0, 1, 5), color = "black", linestyle = "--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc = "lower right")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def ExtendTable(table_output, string_table_column, string_input, string_ps_input, string_input_short, string_data_type):
    # sort the table by run, obs_id and event_id
    table_output = table_output.sort_values(by = ["run", "obs_id", "event_id"])
    table_output.reset_index(drop = True, inplace = True)

    # add empty coloumn to be filled in with CTA images or patter spectra
    table_output[string_table_column] = np.nan
    table_output[string_table_column] = table_output[string_table_column].astype(object)

    # extract unique run ids
    runs_unique = pd.unique(table_output["run"])
    # for loop over output table in order to add CTA image / pattern spectrum to each column
    for n in tqdm(range(len(table_output))): #len(table_output)
        # check if it is an gamma ray or proton
        if table_output["true gammaness"][n] == 1.0:
            particle_type_n = "gamma_diffuse"
        elif table_output["true gammaness"][n] == 0.0:
            particle_type_n = "proton"

        # define the (CNN) input data file name
        run_filename = f"{particle_type_n}_20deg_0deg_run{int(table_output['run'][n])}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
        input_filename = f"dm-finder/cnn/{string_input}/input/{particle_type_n}/" + string_ps_input + run_filename + string_input_short + string_data_type + ".h5"

        # read the (CNN) input data file to extract CTA image / pattern spectrum of correpsoning run, obs_id and event_id
        table_input = pd.read_hdf(input_filename)
        table_input_row = table_input[(table_input["obs_id"] == table_output["obs_id"][n]) & (table_input["event_id"] == table_output["event_id"][n])]

        # add CTA image / pattern spectrum to the table
        table_output[string_table_column][n] = table_input_row.iloc[0][string_table_column]

    return table_output

def PlotWronglyClassifiedEvents(table_output, particle_type, string_table_column, gammaness_limit, path):
    # shuffle the table
    table_output = table_output.sample(frac = 1).reset_index(drop = True)

    for pt in range(len(particle_type)): # gamma + proton
        # extract table of gammas/protons with gammaness limit
        table_gammaness_limit = table_output[(table_output["true gammaness"] == pt) & (table_output["reconstructed gammaness"] >= gammaness_limit[0]) & (table_output["reconstructed gammaness"] <= gammaness_limit[1])]
        table_gammaness_limit.reset_index(drop = True, inplace = True)

        N = 4
        fig, ax = plt.subplots(N, N)
        if particle_type[pt] == "gamma_diffuse":
            fig.suptitle(f"{particle_type[pt]} examples - {gammaness_limit[0]} < gammaness < {gammaness_limit[1]}")
            path_total = path + "_" + particle_type[pt] + f"gl_{gammaness_limit[0]}_{gammaness_limit[1]}" + ".png"
        elif particle_type[pt] == "proton":
            fig.suptitle(f"{particle_type[pt]} examples - {gammaness_limit[2]} < gammaness < {gammaness_limit[3]}")
            path_total = path + "_" + particle_type[pt] + f"gl_{gammaness_limit[2]}_{gammaness_limit[3]}" + ".png"
        ax = ax.ravel()
        for n in range(N**2):
            # ax[n].title.set_text(f"{int(np.round(table_gammaness_limit['E_true / GeV'][n]))} GeV")
            ax[n].imshow(table_gammaness_limit[string_table_column][n], cmap = "Greys_r")
            ax[n].set_xticks([])
            ax[n].set_yticks([])
        plt.tight_layout()
        plt.savefig(path_total, dpi = 250)
        plt.close()

def ExtractPatternSpectraMean(table_output, particle_type, size, gammaness_limit, string_table_column):
    pattern_spectra_mean = np.zeros(shape = (len(particle_type), size[0], size[1]))
    for pt in range(len(particle_type)):
        if particle_type[pt] == "gamma_diffuse":
            table_gammaness_limit = table_output[(table_output["true gammaness"] == pt) & (table_output["reconstructed gammaness"] >= gammaness_limit[0]) & (table_output["reconstructed gammaness"] <= gammaness_limit[1])]
        elif particle_type[pt] == "proton":
            table_gammaness_limit = table_output[(table_output["true gammaness"] == pt) & (table_output["reconstructed gammaness"] >= gammaness_limit[2]) & (table_output["reconstructed gammaness"] <= gammaness_limit[3])]

        pattern_spectra_sum = table_gammaness_limit[string_table_column].sum()
        pattern_spectra_mean[pt] = pattern_spectra_sum / len(table_gammaness_limit)
        # pattern_spectra_sum_normed[pt] = (pattern_spectra_sum - np.min(pattern_spectra_sum))
        # pattern_spectra_mean[pt] = pattern_spectra_mean[pt] / np.max(pattern_spectra_mean[pt])

    return pattern_spectra_mean

def ExtractPatternSpectraMeanGamma(table_output, size, gammaness_limit_gamma, string_table_column):
    pattern_spectra_mean_gamma = np.zeros(shape = (2, size[0], size[1]))

    table_gammaness_limit_correct = table_output[(table_output["true gammaness"] == 1) & (table_output["reconstructed gammaness"] >= gammaness_limit_gamma[0]) & (table_output["reconstructed gammaness"] <= gammaness_limit_gamma[1])]

    table_gammaness_limit_wrong = table_output[(table_output["true gammaness"] == 1) & (table_output["reconstructed gammaness"] >= gammaness_limit_gamma[2]) & (table_output["reconstructed gammaness"] <= gammaness_limit_gamma[3])]

    pattern_spectra_sum_correct = table_gammaness_limit_correct[string_table_column].sum()
    pattern_spectra_sum_wrong = table_gammaness_limit_wrong[string_table_column].sum()

    pattern_spectra_mean_gamma[0] = pattern_spectra_sum_correct / len(table_gammaness_limit_correct)
    pattern_spectra_mean_gamma[1] = pattern_spectra_sum_wrong / len(table_gammaness_limit_wrong)

    return pattern_spectra_mean_gamma    

def ExtractPatternSpectraMeanProton(table_output, size, gammaness_limit_proton, string_table_column):
    pattern_spectra_mean_proton = np.zeros(shape = (2, size[0], size[1]))

    table_gammaness_limit_correct = table_output[(table_output["true gammaness"] == 0) & (table_output["reconstructed gammaness"] >= gammaness_limit_proton[0]) & (table_output["reconstructed gammaness"] <= gammaness_limit_proton[1])]

    table_gammaness_limit_wrong = table_output[(table_output["true gammaness"] == 0) & (table_output["reconstructed gammaness"] >= gammaness_limit_proton[2]) & (table_output["reconstructed gammaness"] <= gammaness_limit_proton[3])]

    pattern_spectra_sum_correct = table_gammaness_limit_correct[string_table_column].sum()
    pattern_spectra_sum_wrong = table_gammaness_limit_wrong[string_table_column].sum()

    pattern_spectra_mean_proton[0] = pattern_spectra_sum_correct / len(table_gammaness_limit_correct)
    pattern_spectra_mean_proton[1] = pattern_spectra_sum_wrong / len(table_gammaness_limit_wrong)

    return pattern_spectra_mean_proton 

def PlotPatternSpectraMean(pattern_spectra_mean, particle_type, attribute, gammaness_limit, cmap, path):
    for pt in range(len(particle_type)):
        plt.figure()
        if particle_type[pt] == "gamma_diffuse":
            plt.title(f"{particle_type[pt]} mean - {gammaness_limit[0]} < gammaness < {gammaness_limit[1]}", fontsize = 12)
            path_total = path + "_" + particle_type[pt] + f"_gl_{gammaness_limit[0]}_{gammaness_limit[1]}" + ".png"
        elif particle_type[pt] == "proton":
            plt.title(f"{particle_type[pt]} mean - {gammaness_limit[2]} < gammaness < {gammaness_limit[3]}", fontsize = 12)
            path_total = path + "_" + particle_type[pt] + f"_gl_{gammaness_limit[2]}_{gammaness_limit[3]}" + ".png"
        plt.imshow(pattern_spectra_mean[pt], cmap = cmap, norm = SymLogNorm(linthresh = 0.1, base = 10))
        # plt.xlabel(f"attribute {attribute[0]}", fontsize = 18)
        # plt.ylabel(f"attribute {attribute[1]}", fontsize = 18)
        plt.xlabel(f"(moment of inertia) / area$^2$", fontsize = 18)
        plt.ylabel(f"area", fontsize = 18)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        cb.set_label(label = "log$_{10}$(flux)", size = 18)
        cb.ax.tick_params(labelsize = 18) 
        plt.tight_layout()
        plt.savefig(path_total, dpi = 250)
        plt.close()

def PlotPatternSpectraDifference(pattern_spectra_mean, particle_type, attributes, gammaness_limit, path):
    # calculate the pattern spectra difference between gamma and proton events
    pattern_spectra_mean_difference = pattern_spectra_mean[0] - pattern_spectra_mean[1]
    pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max = np.min(pattern_spectra_mean_difference), np.max(pattern_spectra_mean_difference)

    if abs(pattern_spectra_mean_difference_min) > abs(pattern_spectra_mean_difference_max):
        pattern_spectra_mean_difference_max = abs(pattern_spectra_mean_difference_min)
    elif abs(pattern_spectra_mean_difference_min) < abs(pattern_spectra_mean_difference_max):
        pattern_spectra_mean_difference_min = - abs(pattern_spectra_mean_difference_max)

    plt.figure()
    plt.title(f"pattern spectra mean difference - {particle_type[0]} - {particle_type[1]}" "\n" f"{gammaness_limit[0]} ({gammaness_limit[2]}) < gammaness < {gammaness_limit[1]} ({gammaness_limit[3]})", fontsize = 10)
    im = plt.imshow(pattern_spectra_mean[0] - pattern_spectra_mean[1], cmap = "RdBu") #, norm = SymLogNorm(linthresh = 0.001, base = 10))
    im.set_clim(pattern_spectra_mean_difference_min, pattern_spectra_mean_difference_max)
    # im.set_clim(-0.09, 0.09)    
    # plt.xlabel(f"attribute {attribute[0]}", fontsize = 18)
    # plt.ylabel(f"attribute {attribute[1]}", fontsize = 18)
    plt.xlabel(f"attribute {attributes[0]}", fontsize = 18)
    plt.ylabel(f"attribute {attributes[1]}", fontsize = 18)
    plt.xticks([])
    plt.yticks([])
    # plt.annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<-", color='black'))
    # plt.annotate('', xy=(-0.1, 1), xycoords='axes fraction', xytext=(-0.1, 0), arrowprops=dict(arrowstyle="<-", color='black'))
    # plt.xlabel(f"a {attributes[0]}", labelpad = 10, fontsize = 18)
    # plt.ylabel(f"a {attributes[1]}", labelpad = 10, fontsize = 18)
    cb = plt.colorbar()
    cb.set_label(label = "log$_{10}$(flux)", size = 18)
    cb.ax.tick_params(labelsize = 18) 
    plt.tight_layout()
    plt.savefig(path + f"_gl_{gammaness_limit[0]}_{gammaness_limit[1]}_{gammaness_limit[2]}_{gammaness_limit[3]}" + ".png", dpi = 250)
    plt.close()