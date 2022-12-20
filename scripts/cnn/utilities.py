import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, MaxPool2D, ReLU, Dropout, LeakyReLU
import matplotlib.patches as mpatches
import sys
from tqdm import tqdm
from matplotlib.colors import SymLogNorm, LogNorm, LinearSegmentedColormap

# np.set_printoptions(threshold=sys.maxsize)
pd.options.mode.chained_assignment = None

plt.rcParams.update({'font.size': 8}) # 8 (paper), 10 (poster)
fontsize_plots = 8 # 8 (paper), 10 (poster)
# plt.rcParams.update({'font.family':'serif'}) #serif
# plt.rcParams["mathtext.fontset"] = 'dejavuserif' #dejavuserif
# pd.options.mode.chained_assignment = None 

plt.rc('text', usetex=True )
plt.rc('font', family='Times New Roman')#, weight='normal', size=14)
plt.rcParams['mathtext.fontset'] = 'cm'

# define some colors and cmaps
color_single = "#143d59"
# colors_categorial = ["#143d59", "#e49d23"] # blue, yellow
colors_categorial = ["#143d59", "#00c6b4"] # blue, turquoise
colors_quat = ["#143d59", "#2C91D2", "#008175", "#00c6b4"] # blue, turquoise
# colors_categorial_hist = ["#143d59", "#93000F"]
colors_categorial_hist = ["#143d59", "#00c6b4"]
cmap_energy_scattering = LinearSegmentedColormap.from_list("", ['#143d59', '#00c6b4', "#fff7d6"])

cm_conversion_factor = 1/2.54  # centimeters in inches
single_column_fig_size = (8.85679 * cm_conversion_factor, 8.85679 * 3/4 * cm_conversion_factor)
single_column_fig_size_legend = (8.85679 * cm_conversion_factor, 8.85679 * 3/4 * 7/6 * cm_conversion_factor)
double_column_fig_size = (18.34621 * cm_conversion_factor, 18.34621 * 3/4 * cm_conversion_factor)
double_column_squeezed_fig_size = (18.34621 * cm_conversion_factor, 18.34621 * 1/2 * cm_conversion_factor)

markers = [".", "s"]
markersizes = [6, 3]
# hatches = ["xxx", "..."]

def cstm_PuBu(x):
    return plt.cm.PuBu((np.clip(x,2,10)-2)/8.)

def cstm_RdBu(x):
    return plt.cm.RdBu((np.clip(x,2,10)-2)/8.)

def PlotLoss(epochs, loss_training, loss_validation, path):
    plt.figure(figsize = single_column_fig_size)
    plt.grid(alpha = 0.2)
    plt.plot(epochs + 1, loss_training, label = "Training", color = colors_categorial[0])
    plt.plot(epochs + 1, loss_validation, label = "Validation", color = colors_categorial[1])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotLossComparison(epochs_all, loss_train_all, loss_val_all, input, path):
    # plot the ROC curve
    plt.figure(figsize = single_column_fig_size_legend)
    plt.grid(alpha = 0.2)
    linestyles = ["-.", "--"]

    for i in range(len(epochs_all)):
        if input[i] == "cta":
            plt.plot(epochs_all[i], loss_train_all[i], linestyle = "solid", color = colors_quat[0], label = "Training (CTA)", alpha = 1.0) 
            plt.plot(epochs_all[i], loss_val_all[i], linestyle = "dashed", color = colors_quat[1], label = "Validation (CTA)", alpha = 1.0) 
        elif input[i] == "ps":
            plt.plot(epochs_all[i], loss_train_all[i], linestyle = "solid", color = colors_quat[2], label = "Training (PS)", alpha = 1.0) 
            plt.plot(epochs_all[i], loss_val_all[i], linestyle = "dashed", color = colors_quat[3], label = "Validation (PS)", alpha = 1.0) 
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(0., 1. , 1., .102), loc="lower left", mode = "expand", ncol = 2)
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotEnergyScattering2D(energy_true, energy_rec, path):
    plt.figure(figsize = single_column_fig_size)
    plt.grid(alpha = 0.2)
    x = np.linspace(np.min(energy_true), np.max(energy_true), 100)
    plt.plot(x, x, color="black", label = "$E_\mathrm{rec} = E_\mathrm{true}$")
    # plt.scatter(x,y,edgecolors='none',s=marker_size,c=void_fraction, norm=matplotlib.colors.LogNorm())
    plt.hist2d(energy_true, energy_rec, bins=(50, 50), cmap = cmap_energy_scattering, norm = matplotlib.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Number of events')
    plt.xlabel("$\log_{10}(E_\mathrm{true}$ $\mathrm{[TeV]})$")
    plt.ylabel("$\log_{10}(E_\mathrm{rec}$ $\mathrm{[TeV]})$")
    plt.legend(loc = "upper left")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotRelativeEnergyError(relative_energy_error_single, mean, sigma_total, path):
    plt.figure()
    plt.grid(alpha = 0.2)
    plt.hist(relative_energy_error_single, bins = np.linspace(np.min(relative_energy_error_single), np.max(relative_energy_error_single), 40), color = color_single)
    # plt.yscale("log")
    plt.xlabel("($E_\mathrm{rec} - E_\mathrm{true})/ E_\mathrm{true}$")
    plt.ylabel("Number of events")
    plt.text(0.95, 0.95, "median $ = %.3f$" % mean, ha = "right", va = "top", transform = plt.gca().transAxes)
    plt.text(0.95, 0.90, "$\sigma = %.3f$" % sigma_total, ha = "right", va = "top", transform = plt.gca().transAxes)
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotRelativeEnergyErrorBinned(energy_true_binned, energy_rec_binned, bins, path):
    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(double_column_squeezed_fig_size)
    ax = ax.ravel()
    for j in range(len(energy_true_binned)):
        relative_energy_error = (energy_rec_binned[j] - energy_true_binned[j]) / energy_true_binned[j] 
        median = np.median(relative_energy_error)
        sigma = np.std(relative_energy_error, ddof = 1)
        ax[j].set_title("{0:.1f} TeV ".format(np.round(bins[j], 1)) + "$< E_\mathrm{true} <$" + " {0:.1f} TeV".format(np.round(bins[j+1], 1)), fontsize = fontsize_plots)
        ax[j].grid(alpha = 0.2)
        ax[j].hist(relative_energy_error, bins = np.linspace(-1, 1, 40), color = color_single)
        ymin, ymax = ax[j].get_ylim()
        ax[j].vlines(median, ymin, ymax, color = "black", linestyle = '-', linewidth = 0.7,  label = "median $ = {0:.3f}$".format(np.round(median, 3)))
        # ax[j].vlines(median - sigma, ymin, ymax, color = "black", linestyle = '--', linewidth = 0.7, label = r"$\sigma = %.3f$" % sigma)
        # ax[j].vlines(median + sigma, ymin, ymax, color = "black", linestyle = '--', linewidth = 0.7)
        ax[j].tick_params(axis='both', which='major')
        ax[j].legend()
        ymax = 1.6 * ymax
        ax[j].set_ylim(ymin, ymax)
    ax[-2].set_xlabel("$\Delta E / E_\mathrm{true}$")
    ax[3].set_ylabel("Number of events")
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
        relative_energy_error_corrected = np.abs((energy_rec_binned[j] - energy_true_binned[j]) / energy_true_binned[j] - median)
        relative_energy_error_corrected = np.sort(relative_energy_error_corrected)
        index_68 = int(len(relative_energy_error_corrected) * 0.68)
        sigma_single = relative_energy_error_corrected[index_68]
        sigma_collection = np.append(sigma_collection, sigma_single)

    return (median_collection, sigma_collection)

  
def PlotRelativeEnergyErrorBinnedCorrected(energy_true_binned, energy_rec_binned, bins, path):
    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(double_column_squeezed_fig_size)
    ax = ax.ravel()
    sigma_collection = np.array([])
    for j in range(len(energy_true_binned)):
        relative_energy_error = (energy_rec_binned[j] - energy_true_binned[j]) / energy_true_binned[j] 
        median = np.median(relative_energy_error)
        relative_energy_error_corrected = np.abs((energy_rec_binned[j] - energy_true_binned[j]) / energy_true_binned[j] - median)
        relative_energy_error_corrected = np.sort(relative_energy_error_corrected)
        index_68 = int(len(relative_energy_error_corrected) * 0.68)

        sigma_single = relative_energy_error_corrected[index_68]
        sigma_collection = np.append(sigma_collection, sigma_single)
        
        ax[j].set_title("{0:.1f} TeV ".format(np.round(bins[j], 1)) + "$< E_\mathrm{true} <$" + " {0:.1f} TeV".format(np.round(bins[j+1], 1)), fontsize = fontsize_plots)
        ax[j].grid(alpha = 0.2)
        ax[j].hist(relative_energy_error_corrected, bins = np.linspace(0, 1, 40), color = color_single)
        ymin, ymax = ax[j].get_ylim()
        ax[j].vlines(sigma_single, ymin, ymax, color = "black", linestyle = '--', linewidth = 0.7,  label = "$\sigma_{68} = %.3f$" % sigma_single)
        ax[j].tick_params(axis='both', which='major', labelsize = fontsize_plots)
        ax[j].legend(fontsize = fontsize_plots)
        ymax = 1.3 * ymax
        ax[j].set_ylim(ymin, ymax)
    ax[-2].set_xlabel("$|\Delta E / E_\mathrm{true}|_{\mathrm{corr}}$")
    ax[3].set_ylabel("Number of events", fontsize = fontsize_plots)
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()


def PlotEnergyAccuracy(median, bins, path):
    bins_central = np.array([])
    for b in range(len(bins) - 1):
        bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    plt.figure()
    plt.grid(alpha = 0.2)
    plt.errorbar(bins_central, median, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".", color = color_single)
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
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
    plt.errorbar(bins_central, sigma, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".", color = color_single)
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
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
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
    plt.ylabel("median$(\Delta E / E_\mathrm{true})$")
    plt.xscale("log")
    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim(ymin, 1.2 * ymax)
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotEnergyAccuracyComparisonMean(median_all, bins, label, args_input, path):
    print("Starting PlotEnergyAccuracyComparisonMean")
    table = []
    for k in range(len(args_input)):
        table.append([args_input[k], median_all[k]])

    table = pd.DataFrame(table, columns=["input", "energy accuracy"])

    table_mean = []
    args_input_unique = np.unique(args_input)
    for k in range(len(args_input_unique)):
        table_k = table.copy()
        table_k.where(table_k["input"] == args_input_unique[k], inplace = True)
        table_mean.append([args_input_unique[k], np.mean(table_k["energy accuracy"].dropna().to_numpy(), axis = 0), np.std(table_k["energy accuracy"].dropna().to_numpy(), axis = 0, ddof = 1)])
    table_mean = pd.DataFrame(table_mean, columns=["input", "mean energy accuracy", "std energy accuracy"])

    bins_central = np.array([])
    for b in range(len(bins) - 1):
        bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    plt.figure(figsize = single_column_fig_size)
    plt.grid(alpha = 0.2)
    plt.plot(np.linspace(bins[0], bins[-1], 10), np.linspace(0, 0, 10), color = "grey", linestyle = "--")
    labels = ["CTA images", "Pattern spectra"]
    for i in range(len(args_input_unique)):
        table_mean_i = table_mean.copy()
        mean_energy_accuracy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean energy accuracy"].dropna().to_numpy()[0]
        std_energy_accuracy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std energy accuracy"].dropna().to_numpy()[0]
        plt.errorbar(bins_central, mean_energy_accuracy, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 0.0, marker = markers[i], markersize = markersizes[i], label = labels[i], color = colors_categorial[i])
        bins_central_fill = np.append(bins_central, bins[-1])
        bins_central_fill = np.insert(bins_central_fill, 0, bins[0])
        filling_lower = mean_energy_accuracy - std_energy_accuracy
        filling_lower = np.append(filling_lower, filling_lower[-1])
        filling_lower = np.insert(filling_lower, 0, filling_lower[0])
        filling_upper = mean_energy_accuracy + std_energy_accuracy
        filling_upper = np.append(filling_upper, filling_upper[-1])
        filling_upper = np.insert(filling_upper, 0, filling_upper[0])
        plt.fill_between(bins_central_fill, filling_lower, filling_upper, facecolor = colors_categorial[i], alpha = 0.3)
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
    plt.ylabel("median$(\Delta E / E_\mathrm{true})$")
    plt.xscale("log")
    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim(ymin, 1.2 * ymax)
    plt.legend()
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
    # plt.fill_between(bins_central, mean - error, mean + error, color = "black", alpha = 0.25)
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
    plt.ylabel("$(\Delta E / E_\mathrm{true})_{68}$")
    plt.xscale("log")
    # plt.ylim(0.2, 0.60)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotEnergyResolutionComparisonMean(args_input, sigma_all, bins, label, path):
    table = []
    for k in range(len(args_input)):
        table.append([args_input[k], sigma_all[k]])

    table = pd.DataFrame(table, columns=["input", "energy resolution"])

    table_mean = []
    args_input_unique = np.unique(args_input)
    for k in range(len(args_input_unique)):
        table_k = table.copy()
        table_k.where(table_k["input"] == args_input_unique[k], inplace = True)
        table_mean.append([args_input_unique[k], np.mean(table_k["energy resolution"].dropna().to_numpy(), axis = 0), np.std(table_k["energy resolution"].dropna().to_numpy(), axis = 0, ddof = 1)])
    table_mean = pd.DataFrame(table_mean, columns=["input", "mean energy resolution", "std energy resolution"])

    bins_central = np.array([])
    for b in range(len(bins) - 1):
        bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    data_cta_requirement = np.genfromtxt("dm-finder/data/South-50h-ERes.dat")
    cta_requirement_energy = 10**data_cta_requirement[:,0]
    cta_requirement_energy_resolution = data_cta_requirement[:,1]

    plt.figure(figsize = single_column_fig_size)
    plt.grid(alpha = 0.2)
    labels = ["CTA images", "Pattern spectra"]
    for i in range(len(args_input_unique)):
        table_mean_i = table_mean.copy()
        mean_energy_resolution = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean energy resolution"].dropna().to_numpy()[0]
        print(mean_energy_resolution)
        std_energy_resolution = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std energy resolution"].dropna().to_numpy()[0]
        plt.errorbar(bins_central, mean_energy_resolution, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 0.0, marker = markers[i], markersize = markersizes[i], label = labels[i], color = colors_categorial[i])
        bins_central_fill = np.append(bins_central, bins[-1])
        bins_central_fill = np.insert(bins_central_fill, 0, bins[0])
        filling_lower = mean_energy_resolution - std_energy_resolution
        filling_lower = np.append(filling_lower, filling_lower[-1])
        filling_lower = np.insert(filling_lower, 0, filling_lower[0])
        filling_upper = mean_energy_resolution + std_energy_resolution
        filling_upper = np.append(filling_upper, filling_upper[-1])
        filling_upper = np.insert(filling_upper, 0, filling_upper[0])
        plt.fill_between(bins_central_fill, filling_lower, filling_upper, facecolor = colors_categorial[i], alpha = 0.3)
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
    plt.ylabel("$(\Delta E / E_\mathrm{true})_{68}$")
    plt.xscale("log")
    # plt.ylim(0.2, 0.60)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.plot(cta_requirement_energy, cta_requirement_energy_resolution, color = "grey", linestyle = "--", label = "CTA requirements")
    plt.xlim(xmin, xmax)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = "upper left")
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

    plt.figure(figsize = single_column_fig_size)
    plt.grid(alpha = 0.2)
    plt.hist(gammaness_gammas, label = "True photons", bins = np.linspace(0, 1, 26), alpha = 0.8, color = colors_categorial_hist[0])
    plt.hist(gammaness_protons, label = "True protons", bins = np.linspace(0, 1, 26), alpha = 0.8, color = colors_categorial_hist[1])
    # plt.axvline(0.5, color = "r", linestyle = "--", label = "decision boundary")
    plt.xlabel("Gammaness")
    plt.ylabel("Number events")
    # plt.yscale("log")
    # handles, labels = fig.gca().get_legend_handles_labels()
    # patch = mpatches.Patch(color = 'white', label = r"total - acc = {0} $\%$".format(np.round(true_positive_rate_total * 100, 2)))
    # handles.append(patch) 
    # plt.legend(handles = handles, framealpha = 0.95)
    plt.legend(framealpha = 0.95, loc = "upper center")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def tpr(x, thresholds):
    true_positive_rate = np.array([])
    for i in range(len(thresholds)):
        true_positive_rate = np.append(true_positive_rate, len(x[x >= thresholds[i]]) / len(x))

    return(true_positive_rate)

def fpr(x, thresholds):
    false_positive_rate = np.array([])
    for i in range(len(thresholds)):
        false_positive_rate = np.append(false_positive_rate, len(x[x >= thresholds[i]]) / len(x))

    return(false_positive_rate)

def PositiveRates(x, y, thresholds): # x = gammaness_gammas, y = gammaness_protons, thresholds = gammaness thresholds
    true_positive_rate = np.array([])
    false_positive_rate = np.array([])

    for i in range(len(thresholds)):
        true_positive_rate = np.append(true_positive_rate, len(x[x >= thresholds[i]]) / len(x))
        false_positive_rate = np.append(false_positive_rate, len(y[y >= thresholds[i]]) / len(y))

    return(true_positive_rate, false_positive_rate)

def NegativeRates(x, y, thresholds): # x = gammaness_gammas, y = gammaness_protons, thresholds = gammaness thresholds
    true_negative_rate = np.array([])
    false_negative_rate = np.array([])

    for i in range(len(thresholds)):
        true_negative_rate = np.append(true_negative_rate, len(y[y < thresholds[i]]) / len(y))
        false_negative_rate = np.append(false_negative_rate, len(x[x < thresholds[i]]) / len(x))

    return(true_negative_rate, false_negative_rate)

# area under ROC curve (Riemann integral)
def AreaUnderROCCurve(x, y): # x = gammaness_gammas, y = gammaness_protons
    x = x[::-1]
    y = y[::-1]
    area_under_ROC_curve = np.sum(y[1:] * np.diff(x))
    return area_under_ROC_curve

# def AreaUnderROCCurveNumpy(x, y):
#     area = np.trapz(y[::-1], x[::-1])
#     return area

def PlotGammanessEnergyBinned(table_output, energy_range, path):
    table = table_output.copy()
    table["E_true / GeV"] = table["E_true / GeV"] * 1e-3
    table.columns = table.columns.str.replace("E_true / GeV", "E_true / TeV")

    true_gammas = table.where(table["true gammaness"] == 1).dropna().reset_index()
    true_protons = table.where(table["true gammaness"] == 0).dropna().reset_index()

    gammaness_gammas = true_gammas["reconstructed gammaness"].to_numpy()
    gammaness_protons = true_protons["reconstructed gammaness"].to_numpy()

    number_energy_ranges = 9 # number of energy ranges the whole energy range will be splitted
    sst_energy_min = energy_range[0] # TeV
    sst_energy_max = energy_range[1] # TeV
    bins = np.logspace(np.log10(np.min(sst_energy_min)), np.log10(np.max(sst_energy_max)), number_energy_ranges + 1) 

    bins_central = np.array([])
    for b in range(len(bins) - 1):
        bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(double_column_squeezed_fig_size)
    ax = ax.ravel()
    plt.grid(alpha = 0.2)
    for i in range(number_energy_ranges):
        ax[i].set_title("{0:.1f} - {1:.1f} TeV".format(bins[i], bins[i+1]))
        gammaness_gammas_binned = true_gammas[(true_gammas["E_true / TeV"] >= bins[i]) & (true_gammas["E_true / TeV"] <= bins[i+1])]["reconstructed gammaness"].to_numpy()
        gammaness_protons_binned = true_protons[(true_protons["E_true / TeV"] >= bins[i]) & (true_protons["E_true / TeV"] <= bins[i+1])]["reconstructed gammaness"].to_numpy()

        hist_gamma = ax[i].hist(gammaness_gammas_binned, bins = np.linspace(0, 1, 31), alpha = 0.5, color = colors_categorial_hist[0])
        ax[i].hist(gammaness_protons_binned, bins = np.linspace(0, 1, 31), alpha = 0.5, color = colors_categorial_hist[1])
        x = np.array([])
        for k in range(len(hist_gamma[1]) - 1):
            x = np.append(x, (hist_gamma[1][k+1] - hist_gamma[1][k]) / 8 + hist_gamma[1][k])
        for m, n in zip(x, hist_gamma[0]):
            ax[i].text(m, 1.1 * n, str(int(n)), color = "black", fontweight = "bold", rotation= 90, fontsize = 5)

        # create gammaness energy binnded plots
        ax[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : fontsize_plots})
        ax[i].set_xlim(-0.05, 1.05)
        ax[i].tick_params(axis = 'both', which = 'major')
    ax[-2].set_xlabel("Gammaness")
    ax[3].set_ylabel("Number events")
    # plt.legend(framealpha = 0.95, fontsize = 10)
    plt.tight_layout()
    plt.savefig(path + "gammaness_energy_binned.pdf", dpi = 250)
    plt.close()


    # ######################################################################

    # # prepare energy binning
    # table_output = table_output.sort_values(by = ["E_true / GeV"], ignore_index = True)
    # energy_true = np.asarray(table_output["E_true / GeV"]) * 1e-3
    # gammaness_true = np.asarray(table_output["true gammaness"])
    # gammaness_rec = np.asarray(table_output["reconstructed gammaness"])

    # number_energy_ranges = 9 # number of energy ranges the whole energy range will be splitted
    # sst_energy_min = energy_range[0] # TeV
    # sst_energy_max = energy_range[1] # TeV
    # bins = np.logspace(np.log10(np.min(sst_energy_min)), np.log10(np.max(sst_energy_max)), number_energy_ranges + 1) 
    # indices = np.array([], dtype = int)
    # for b in range(len(bins) - 2):
    #     index = np.max(np.where(energy_true < bins[b+1])) + 1
    #     indices = np.append(indices, index)

    # energy_true_binned = np.split(energy_true, indices)
    # gammaness_true_binned = np.split(gammaness_true, indices)
    # gammaness_rec_binned = np.split(gammaness_rec, indices)

    # fig, ax = plt.subplots(3, 3)
    # fig.set_size_inches(double_column_squeezed_fig_size)
    # ax = ax.ravel()
    # plt.grid(alpha = 0.2)
    # accuracy_energy = np.array([])
    # area_under_ROC_curve_energy = np.array([])
    # true_positive_rate_energy = np.array([])
    # false_positive_rate_energy = np.array([])
    # for i in range(number_energy_ranges):
    #     # define true gammaness as boolean
    #     gammaness_true_bool = gammaness_true_binned[i].astype(bool)
    #     gammaness_true_bool_inverted = [not elem for elem in gammaness_true_bool]

    #     # extract gammaness of true gamma-ray events
    #     gammaness_gammas = gammaness_rec_binned[i][gammaness_true_bool]
    #     print("GammanessEnergyBinned")
    #     print("bin", bins[i], bins[i+1])
    #     print("gammaness_gammas", gammaness_gammas)
    #     # extract gammaness of true proton events
    #     gammaness_protons = gammaness_rec_binned[i][gammaness_true_bool_inverted]
    #     print("gammaness_protons", gammaness_protons)

    #     # calculate the true positive rate for gamma rays and protons depending on the threshold
    #     thresholds = np.linspace(0, 1.0, 9999)
    #     # calculate rates to determine AUC
    #     true_positive_rate, false_positive_rate = PositiveRates(gammaness_gammas, gammaness_protons, thresholds)
    #     true_negative_rate, false_negative_rate = NegativeRates(gammaness_gammas, gammaness_protons, thresholds)
    #     area_under_ROC_curve = AreaUnderROCCurve(false_positive_rate, true_positive_rate)
    #     area_under_ROC_curve_energy = np.append(area_under_ROC_curve_energy, area_under_ROC_curve)

    #     # determine accuracy for gammaness == 0.7
    #     gammaness_cut = 0.7
    #     accuracy = (len(gammaness_gammas[gammaness_gammas >= gammaness_cut]) + len(gammaness_protons[gammaness_protons < gammaness_cut])) / (len(gammaness_gammas) + len(gammaness_protons))
    #     accuracy_energy = np.append(accuracy_energy, accuracy)

    #     # determine the efficiencies (TPR,...) for gammaness == 0.7
    #     true_positive_rate_single, false_positive_rate_single = PositiveRates(gammaness_gammas, gammaness_protons, np.array([gammaness_cut]))
    #     true_negative_rate_single, false_negative_rate_single = NegativeRates(gammaness_gammas, gammaness_protons, np.array([gammaness_cut]))
    #     true_positive_rate_energy = np.append(true_positive_rate_energy, true_positive_rate_single)
    #     false_positive_rate_energy = np.append(false_positive_rate_energy, false_positive_rate_single)

    #     # create gammaness energy binnded plots
    #     ax[i].set_title(f"{np.round(bins[i], 1)} - {np.round(bins[i+1], 1)} TeV", fontdict = {"fontsize" : fontsize_plots})
    #     ax[i].hist(gammaness_gammas, label = "True photons", bins = np.linspace(0, 1, 31), alpha = 0.8, color = colors_categorial_hist[0])
    #     ax[i].hist(gammaness_protons, label = "True protons", bins = np.linspace(0, 1, 31), alpha = 0.8, color = colors_categorial_hist[1])
    #     ylim = ax[i].get_ylim()
    #     ax[i].set_xlim(-0.05, 1.05)
    #     # ax[i].text(0, 0.8 * ylim[1], f"AUC = {np.round(area_under_ROC_curve, 3)}", fontsize = 6)
    #     # ax[i].axvline(0.5, color = "r", linestyle = "--", label = "decision boundary")
    #     ax[i].tick_params(axis = 'both', which = 'major')

    # #ax[-1].set_xlabel("Gammaness", fontsize = 10)
    # ax[-2].set_xlabel("Gammaness")
    # ax[3].set_ylabel("Number events")
    # # plt.legend(framealpha = 0.95, fontsize = 10)
    # plt.tight_layout()
    # plt.savefig(path + "gammaness_energy_binned.pdf", dpi = 250)
    # plt.close()

    # bins_central = np.array([])
    # for b in range(len(bins) - 1):
    #     bins_central = np.append(bins_central, bins[b] + (bins[b+1] - bins[b]) / 2)

    # plt.figure()
    # plt.grid(alpha = 0.2)
    # plt.errorbar(bins_central, area_under_ROC_curve_energy, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".", color = color_single)
    # plt.xlabel("$E_\mathrm{true}$ [TeV]")
    # plt.ylabel("AUC")
    # plt.xscale("log")
    # plt.tight_layout()
    # plt.savefig(path + "AUC_energy.pdf", dpi = 250)
    # plt.close()

    # plt.figure()
    # plt.grid(alpha = 0.2)
    # plt.errorbar(bins_central, accuracy_energy, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".", color = color_single)
    # plt.xlabel("$E_\mathrm{true}$ [TeV]")
    # plt.ylabel("Accuracy")
    # plt.xscale("log")
    # plt.tight_layout()
    # plt.savefig(path + "accuracy_energy.pdf", dpi = 250)
    # plt.close()

    # plt.figure()
    # plt.grid(alpha = 0.2)
    # plt.errorbar(bins_central, true_positive_rate_energy, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".", color = colors_categorial[0], label = "Photon")
    # plt.errorbar(bins_central, false_positive_rate_energy, xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 3.0, marker = ".", color = colors_categorial[1], label = "Proton")
    # plt.xlabel("$E_\mathrm{true}$ [TeV]")
    # plt.ylabel(r"$\eta$")
    # plt.xscale("log")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(path + "efficiencies_energy.pdf", dpi = 250)
    # plt.close()

    return(bins, bins_central)

# def GetEfficienciesEnergyBinned(table_output, energy_range_gamma, energy_range_proton):
#     # prepare energy binning
#     table_output = table_output.sort_values(by = ["E_true / GeV"], ignore_index = True)
#     energy_true = np.asarray(table_output["E_true / GeV"]) * 1e-3
#     gammaness_true = np.asarray(table_output["true gammaness"])
#     gammaness_rec = np.asarray(table_output["reconstructed gammaness"])

#     number_energy_ranges_gamma = 11 # number of energy ranges the whole energy range will be splitted
#     number_energy_ranges_proton = 9
#     gamma_energy_min, gamma_energy_max = energy_range_gamma[0], energy_range_gamma[1] # TeV
#     proton_energy_min, proton_energy_max = energy_range_proton[0], energy_range_proton[1] # TeV
#     bins_gamma = np.logspace(np.log10(np.min(gamma_energy_min)), np.log10(np.max(gamma_energy_max)), number_energy_ranges_gamma + 1) 
#     bins_proton = np.logspace(np.log10(np.min(proton_energy_min)), np.log10(np.max(proton_energy_max)), number_energy_ranges_proton + 1) 

#     indices_gamma = np.array([], dtype = int)
#     indices_proton = np.array([], dtype = int)
#     for b in range(len(bins_gamma) - 2):
#         index_gamma = np.max(np.where(energy_true < bins_gamma[b+1])) + 1
#         indices_gamma = np.append(indices_gamma, index_gamma)
    
#     for b in range(len(bins_proton) - 2):
#         index_proton = np.max(np.where(energy_true < bins_proton[b+1])) + 1
#         indices_proton = np.append(indices_proton, index_proton)

#     energy_true_binned_gamma = np.split(energy_true, indices_gamma)
#     energy_true_binned_proton = np.split(energy_true, indices_proton)

#     gammaness_true_binned_er_gamma = np.split(gammaness_true, indices_gamma)
#     gammaness_true_binned_er_proton = np.split(gammaness_true, indices_proton)
#     gammaness_rec_binned_er_gamma = np.split(gammaness_rec, indices_gamma)
#     gammaness_rec_binned_er_proton = np.split(gammaness_rec, indices_proton)

#     area_under_ROC_curve_energy = np.array([])
#     true_positive_rate_er_gamma = np.array([])
#     false_positive_rate_er_proton = np.array([])
#     for i in range(number_energy_ranges_gamma):
#         # define true gammaness as boolean
#         gammaness_true_bool_er_gamma = gammaness_true_binned_er_gamma[i].astype(bool)
#         gammaness_true_bool_er_gamma_inverted = [not elem for elem in gammaness_true_bool_er_gamma]

#         # extract gammaness of true gamma-ray events
#         gammaness_gammas_er_gamma = gammaness_rec_binned_er_gamma[i][gammaness_true_bool_er_gamma]
#         # extract gammaness of true proton events

#         # calculate the true positive rate for gamma rays and protons depending on the threshold
#         thresholds = np.linspace(0, 1.0, 9999)

#         # determine accuracy for gammaness == 0.7
#         gammaness_cut = 0.7

#         # determine the efficiencies (TPR,...) for gammaness == 0.7
#         true_positive_rate_er_gamma_single = tpr(gammaness_gammas_er_gamma, np.array([gammaness_cut]))

#         true_positive_rate_er_gamma = np.append(true_positive_rate_er_gamma, true_positive_rate_er_gamma_single)

#     for i in range(number_energy_ranges_proton):
#         # define true gammaness as boolean
#         gammaness_true_bool_er_proton = gammaness_true_binned_er_proton[i].astype(bool)
#         gammaness_true_bool_er_proton_inverted = [not elem for elem in gammaness_true_bool_er_proton]

#         # extract gammaness of true proton events
#         gammaness_protons_er_proton = gammaness_rec_binned_er_proton[i][gammaness_true_bool_er_proton_inverted]

#         # calculate the true positive rate for gamma rays and protons depending on the threshold
#         thresholds = np.linspace(0, 1.0, 9999)
        
#         # determine accuracy for gammaness == 0.7
#         gammaness_cut = 0.7

#         # determine the efficiencies (TPR,...) for gammaness == 0.7
#         false_positive_rate_er_proton_single = fpr(gammaness_protons_er_proton, np.array([gammaness_cut]))

#         false_positive_rate_er_proton = np.append(false_positive_rate_er_proton, false_positive_rate_er_proton_single)

#     bins_central_gamma = np.array([])
#     for b in range(len(bins_gamma) - 1):
#         bins_central_gamma = np.append(bins_central_gamma, bins_gamma[b] + (bins_gamma[b+1] - bins_gamma[b]) / 2)

#     bins_central_proton = np.array([])
#     for b in range(len(bins_proton) - 1):
#         bins_central_proton = np.append(bins_central_proton, bins_proton[b] + (bins_proton[b+1] - bins_proton[b]) / 2)

#     return(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, true_positive_rate_er_gamma, false_positive_rate_er_proton)

# def GetEfficienciesEnergyBinned(table_output, energy_range_gamma, energy_range_proton):
#     # prepare energy binning
#     table_output = table_output.sort_values(by = ["E_true / GeV"], ignore_index = True)
#     energy_true = np.asarray(table_output["E_true / GeV"]) * 1e-3
#     gammaness_true = np.asarray(table_output["true gammaness"])
#     gammaness_rec = np.asarray(table_output["reconstructed gammaness"])

#     number_energy_ranges_gamma = 9 # 11 # number of energy ranges the whole energy range will be splitted
#     number_energy_ranges_proton = 9
#     gamma_energy_min, gamma_energy_max = energy_range_gamma[0], energy_range_gamma[1] # TeV
#     proton_energy_min, proton_energy_max = energy_range_proton[0], energy_range_proton[1] # TeV
#     bins_gamma = np.logspace(np.log10(np.min(gamma_energy_min)), np.log10(np.max(gamma_energy_max)), number_energy_ranges_gamma + 1) 
#     bins_proton = np.logspace(np.log10(np.min(proton_energy_min)), np.log10(np.max(proton_energy_max)), number_energy_ranges_proton + 1) 

#     indices_gamma = np.array([], dtype = int)
#     indices_proton = np.array([], dtype = int)
#     for b in range(len(bins_gamma) - 2):
#         index_gamma = np.max(np.where(energy_true < bins_gamma[b+1])) + 1
#         indices_gamma = np.append(indices_gamma, index_gamma)
    
#     for b in range(len(bins_proton) - 2):
#         index_proton = np.max(np.where(energy_true < bins_proton[b+1])) + 1
#         indices_proton = np.append(indices_proton, index_proton)

#     energy_true_binned_gamma = np.split(energy_true, indices_gamma)
#     energy_true_binned_proton = np.split(energy_true, indices_proton)

#     gammaness_true_binned_er_gamma = np.split(gammaness_true, indices_gamma)
#     gammaness_true_binned_er_proton = np.split(gammaness_true, indices_proton)
#     gammaness_rec_binned_er_gamma = np.split(gammaness_rec, indices_gamma)
#     gammaness_rec_binned_er_proton = np.split(gammaness_rec, indices_proton)

#     area_under_ROC_curve_energy = np.array([])
#     true_positive_rate_er_gamma = np.array([])
#     false_positive_rate_er_proton = np.array([])
#     gammaness_cut_fpr_requirement_binned = np.array([])

#     for i in range(number_energy_ranges_proton):
#         # define true gammaness as boolean
#         gammaness_true_bool_er_proton = gammaness_true_binned_er_proton[i].astype(bool)
#         gammaness_true_bool_er_proton_inverted = [not elem for elem in gammaness_true_bool_er_proton]

#         # extract gammaness of true proton events
#         gammaness_protons_er_proton = gammaness_rec_binned_er_proton[i][gammaness_true_bool_er_proton_inverted]

#         # calculate the true positive rate for gamma rays and protons depending on the threshold
#         thresholds = np.linspace(0, 1.0, 9999)
        
#         # determine accuracy for gammaness == 0.7
#         gammaness_cut = np.linspace(0, 1.0, 1000)
#         false_positive_rate_requirement = 0.01

#         # determine the efficiencies (TPR,...) for gammaness == 0.7
#         false_positive_rate_er_proton_single = fpr(gammaness_protons_er_proton, np.array(gammaness_cut))
#         index = np.argmin(np.abs(false_positive_rate_requirement - false_positive_rate_er_proton_single))
#         gammaness_cut_fpr_requirement = gammaness_cut[index]
#         gammaness_cut_fpr_requirement_binned = np.append(gammaness_cut_fpr_requirement_binned, gammaness_cut_fpr_requirement)

#         false_positive_rate_er_proton = np.append(false_positive_rate_er_proton, false_positive_rate_er_proton_single[index])

#     for i in range(number_energy_ranges_gamma - number_energy_ranges_proton):
#         gammaness_cut_fpr_requirement_binned = np.insert(gammaness_cut_fpr_requirement_binned, 0, gammaness_cut_fpr_requirement_binned[0])

#     for i in range(number_energy_ranges_gamma):
#         # define true gammaness as boolean
#         gammaness_true_bool_er_gamma = gammaness_true_binned_er_gamma[i].astype(bool)
#         gammaness_true_bool_er_gamma_inverted = [not elem for elem in gammaness_true_bool_er_gamma]

#         # extract gammaness of true gamma-ray events
#         gammaness_gammas_er_gamma = gammaness_rec_binned_er_gamma[i][gammaness_true_bool_er_gamma]
#         # extract gammaness of true proton events

#         # calculate the true positive rate for gamma rays and protons depending on the threshold
#         thresholds = np.linspace(0, 1.0, 9999)

#         # # determine accuracy for gammaness == 0.7
#         # gammaness_cut = 0.7

#         # determine the efficiencies (TPR,...) for gammaness == 0.7
#         true_positive_rate_er_gamma_single = tpr(gammaness_gammas_er_gamma, np.array([gammaness_cut_fpr_requirement_binned[i]]))

#         true_positive_rate_er_gamma = np.append(true_positive_rate_er_gamma, true_positive_rate_er_gamma_single)

#     bins_central_gamma = np.array([])
#     for b in range(len(bins_gamma) - 1):
#         bins_central_gamma = np.append(bins_central_gamma, bins_gamma[b] + (bins_gamma[b+1] - bins_gamma[b]) / 2)

#     bins_central_proton = np.array([])
#     for b in range(len(bins_proton) - 1):
#         bins_central_proton = np.append(bins_central_proton, bins_proton[b] + (bins_proton[b+1] - bins_proton[b]) / 2)

#     return(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, true_positive_rate_er_gamma, false_positive_rate_er_proton)

def GetEfficienciesEnergyBinned(table_output, energy_range_gamma, energy_range_proton):
    table = table_output.copy()
    table["E_true / GeV"] = table["E_true / GeV"] * 1e-3
    table.columns = table.columns.str.replace("E_true / GeV", "E_true / TeV")

    true_gammas = table.where(table["true gammaness"] == 1).dropna().reset_index()
    true_protons = table.where(table["true gammaness"] == 0).dropna().reset_index()

    gammaness_gammas = true_gammas["reconstructed gammaness"].to_numpy()
    gammaness_protons = true_protons["reconstructed gammaness"].to_numpy()

    number_energy_ranges_gamma = 9 # 11 # number of energy ranges the whole energy range will be splitted
    number_energy_ranges_proton = 9
    gamma_energy_min, gamma_energy_max = energy_range_gamma[0], energy_range_gamma[1] # TeV
    proton_energy_min, proton_energy_max = energy_range_proton[0], energy_range_proton[1] # TeV
    bins_gamma = np.logspace(np.log10(gamma_energy_min), np.log10(gamma_energy_max), number_energy_ranges_gamma + 1) 
    bins_proton = np.logspace(np.log10(proton_energy_min), np.log10(proton_energy_max), number_energy_ranges_proton + 1) 

    bins_central_gamma = np.array([])
    for b in range(len(bins_gamma) - 1):
        bins_central_gamma = np.append(bins_central_gamma, bins_gamma[b] + (bins_gamma[b+1] - bins_gamma[b]) / 2)

    bins_central_proton = np.array([])
    for b in range(len(bins_proton) - 1):
        bins_central_proton = np.append(bins_central_proton, bins_proton[b] + (bins_proton[b+1] - bins_proton[b]) / 2)

    true_positive_rate_er_gamma = np.array([])
    false_positive_rate_er_proton = np.array([])
    gammaness_cut_fpr_requirement_binned = np.array([])
    for i in range(number_energy_ranges_proton):
        # gammaness_gammas_binned = true_gammas[(true_gammas["E_true / TeV"] >= bins[i]) & (true_gammas["E_true / TeV"] <= bins[i+1])]["reconstructed gammaness"].to_numpy()
        gammaness_protons_er_proton = true_protons[(true_protons["E_true / TeV"] >= bins_proton[i]) & (true_protons["E_true / TeV"] <= bins_proton[i+1])]["reconstructed gammaness"].to_numpy()

        # calculate the true positive rate for gamma rays and protons depending on the threshold
        gammaness_cut = np.linspace(0, 1.0, 9999)
        false_positive_rate_requirement = 0.01

        # determine the efficiencies (TPR,...) for gammaness == 0.7
        false_positive_rate_er_proton_single = fpr(gammaness_protons_er_proton, gammaness_cut)
        index = np.argmin(np.abs(false_positive_rate_requirement - false_positive_rate_er_proton_single))
        gammaness_cut_fpr_requirement = gammaness_cut[index]
        gammaness_cut_fpr_requirement_binned = np.append(gammaness_cut_fpr_requirement_binned, gammaness_cut_fpr_requirement)

        false_positive_rate_er_proton = np.append(false_positive_rate_er_proton, false_positive_rate_er_proton_single[index])
        # false_positive_rate_er_proton = np.append(false_positive_rate_er_proton, fpr(gammaness_protons_er_proton, np.array([0.8])))

    for i in range(number_energy_ranges_gamma - number_energy_ranges_proton):
        gammaness_cut_fpr_requirement_binned = np.insert(gammaness_cut_fpr_requirement_binned, 0, gammaness_cut_fpr_requirement_binned[0])

    for i in range(number_energy_ranges_gamma):
        # gammaness_gammas_binned = true_gammas[(true_gammas["E_true / TeV"] >= bins[i]) & (true_gammas["E_true / TeV"] <= bins[i+1])]["reconstructed gammaness"].to_numpy()
        gammaness_gammas_er_gamma = true_gammas[(true_gammas["E_true / TeV"] >= bins_gamma[i]) & (true_gammas["E_true / TeV"] <= bins_gamma[i+1])]["reconstructed gammaness"].to_numpy()
        
        true_positive_rate_er_gamma_single = tpr(gammaness_gammas_er_gamma, np.array([gammaness_cut_fpr_requirement_binned[i]]))
        # true_positive_rate_er_gamma_single = tpr(gammaness_gammas_er_gamma, np.array([0.8]))

        true_positive_rate_er_gamma = np.append(true_positive_rate_er_gamma, true_positive_rate_er_gamma_single)

    return(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, true_positive_rate_er_gamma, false_positive_rate_er_proton)


def PlotEfficienciesEnergyBinned(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, true_positive_rate_er_gamma, false_positive_rate_er_proton, path):
    plt.figure()
    plt.grid(alpha = 0.2)
    plt.errorbar(bins_central_gamma, true_positive_rate_er_gamma, xerr = (bins_gamma[:-1] - bins_central_gamma, bins_central_gamma - bins_gamma[1:]), linestyle = "", capsize = 3.0, marker = ".", color = colors_categorial[0], label = "Photon")
    plt.errorbar(bins_central_proton, false_positive_rate_er_proton, xerr = (bins_proton[:-1] - bins_central_proton, bins_central_proton - bins_proton[1:]), linestyle = "", capsize = 3.0, marker = ".", color = colors_categorial[1], label = "Proton")
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
    plt.ylabel(r"$\eta$")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def ROC(gammaness_true, gammaness_rec):
     # define true gammaness as boolean
    gammaness_true_bool = gammaness_true.astype(bool)
    gammaness_true_bool_inverted = [not elem for elem in gammaness_true_bool]
    # extract gammaness of true gamma-ray events
    gammaness_gammas = gammaness_rec[gammaness_true_bool]
    # extract gammaness of true proton events
    gammaness_protons = gammaness_rec[gammaness_true_bool_inverted]

    # calculate the true positive rate for gamma rays and protons depending on the threshold
    thresholds = np.linspace(0, 1.0, 9999)
    true_positive_rate, false_positive_rate = PositiveRates(gammaness_gammas, gammaness_protons, thresholds)
    true_negative_rate, false_negative_rate = NegativeRates(gammaness_gammas, gammaness_protons, thresholds)
    true_positive_rate_50, false_positive_rate_50 = PositiveRates(gammaness_gammas, gammaness_protons, np.array([0.5]))
    true_negative_rate_50, false_negative_rate_50 = NegativeRates(gammaness_gammas, gammaness_protons, np.array([0.5]))
    # rejection_power = np.divide(1, false_positive_rate)
    rejection_power = 0 # placeholder
    # rejection_power = 1 / true_negative_rate[true_negative_rate == 0] = None
    # print("Correctly classified photons: ", true_positive_rate_50[0])
    # print("Correctly classified protons: ", true_negative_rate_50[0])
    area_under_ROC_curve = AreaUnderROCCurve(false_positive_rate, true_positive_rate)

    return(true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate, rejection_power, area_under_ROC_curve)


def PlotROC(true_positive_rate, false_positive_rate, area_under_ROC_curve, path):
    # plot the ROC curve
    plt.figure()
    plt.grid(alpha = 0.2)
    plt.plot(false_positive_rate, true_positive_rate, label = "AUC = {0:.3f}".format(np.round(area_under_ROC_curve, 3)), color = color_single) # \nCC $\gamma$ = {1:.3f}\nCC $p$ = {2:.3f}.format(np.round(area_under_ROC_curve, 3), np.round(true_positive_rate_50[0], 3), np.round(true_negative_rate_50[0], 3)))
    plt.plot(np.linspace(0, 1, 5), np.linspace(0, 1, 5), color = "black", linestyle = "--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc = "lower right")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def SaveROC(true_positive_rate, false_positive_rate, area_under_ROC_curve, path):
    table = pd.DataFrame()
    table["FPR"] = false_positive_rate
    table["TPR"] = true_positive_rate
    table["AUC"] = area_under_ROC_curve
    table.to_csv(path)

def PlotPurityGammaness(true_positive_rate, false_positive_rate, rejection_power, thresholds, path):
    # plot the "efficiency" curve
    plt.figure()
    plt.grid(alpha = 0.2)
    plt.plot(thresholds, true_positive_rate, label = "TPR", color = colors_categorial[0]) 
    plt.plot(thresholds, false_positive_rate, label = "FPR", color = colors_categorial[1]) 
    # plt.plot(thresholds, rejection_power, label = "RP", color = "black") 
    plt.xlabel("Gammaness")
    plt.ylabel(r"$\eta$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def AccuracyGammaness(gammaness_true, gammaness_rec):
    # define true gammaness as boolean
    gammaness_true_bool = gammaness_true.astype(bool)
    gammaness_true_bool_inverted = [not elem for elem in gammaness_true_bool]
    # extract gammaness of true gamma-ray events
    gammaness_gammas = gammaness_rec[gammaness_true_bool]
    # extract gammaness of true proton events
    gammaness_protons = gammaness_rec[gammaness_true_bool_inverted]

    # calculate the true positive rate for gamma rays and protons depending on the threshold
    thresholds = np.linspace(0, 1.0, 9999)

    # create empty array for accuracy gammaness to be filled in
    accuracy_gammaness = np.array([])
    for i in range(len(thresholds)):
        # calculate accuracy (TP+TN) / (P+N) for each gammaness threshold
        accuracy_gammaness_i = (len(gammaness_gammas[gammaness_gammas >= thresholds[i]]) + len(gammaness_protons[gammaness_protons < thresholds[i]])) / (len(gammaness_gammas) + len(gammaness_protons))
        # fill empty array with values
        accuracy_gammaness = np.append(accuracy_gammaness, accuracy_gammaness_i)
    
    return(accuracy_gammaness, thresholds)

def PlotAccuracyGammaness(accuracy_gammaness, thresholds, path):
    plt.figure()
    plt.grid(alpha = 0.2)
    plt.plot(thresholds, accuracy_gammaness, color = color_single) # \nCC $\gamma$ = {1:.3f}\nCC $p$ = {2:.3f}.format(np.round(area_under_ROC_curve, 3), np.round(true_positive_rate_50[0], 3), np.round(true_negative_rate_50[0], 3)))
    plt.xlabel("Gammaness")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PrecisionGammaness(gammaness_true, gammaness_rec):
    # define true gammaness as boolean
    gammaness_true_bool = gammaness_true.astype(bool)
    gammaness_true_bool_inverted = [not elem for elem in gammaness_true_bool]
    # extract gammaness of true gamma-ray events
    gammaness_gammas = gammaness_rec[gammaness_true_bool]
    # extract gammaness of true proton events
    gammaness_protons = gammaness_rec[gammaness_true_bool_inverted]

    # calculate the true positive rate for gamma rays and protons depending on the threshold
    thresholds = np.linspace(0, 1.0, 9999)

    # create empty array for precision gammaness to be filled in
    precision_gammaness = np.array([])

    for i in range(len(thresholds)):
        if len(gammaness_gammas[gammaness_gammas >= thresholds[i]]) + len(gammaness_protons[gammaness_protons >= thresholds[i]]) != 0: # to avoid devision by zero
            precision_gammaness_i = len(gammaness_gammas[gammaness_gammas >= thresholds[i]]) / (len(gammaness_gammas[gammaness_gammas >= thresholds[i]]) + len(gammaness_protons[gammaness_protons >= thresholds[i]]))
            precision_gammaness = np.append(precision_gammaness, precision_gammaness_i)
            threshold_cut = i + 1
    
    return(precision_gammaness, threshold_cut)

def PlotPrecisionGammaness(precision_gammaness, thresholds, path):
    plt.figure()
    plt.grid(alpha = 0.2)
    plt.plot(thresholds, precision_gammaness, color = color_single) # \nCC $\gamma$ = {1:.3f}\nCC $p$ = {2:.3f}.format(np.round(area_under_ROC_curve, 3), np.round(true_positive_rate_50[0], 3), np.round(true_negative_rate_50[0], 3)))
    plt.xlabel("Gammaness")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotROCComparison(true_positive_rate_all, false_positive_rate_all, area_under_ROC_curve_all, args_input, path):
    table = []
    for k in range(len(args_input)):
        table.append([args_input[k], true_positive_rate_all[k], false_positive_rate_all[k], area_under_ROC_curve_all[k]])

    table = pd.DataFrame(table, columns=["input", "TPR", "FPR", "AUC"])

    table_mean = []
    args_input_unique = np.unique(args_input)
    for k in range(len(args_input_unique)):
        table_k = table.copy()
        table_k.where(table_k["input"] == args_input_unique[k], inplace = True)
        table_mean.append([args_input_unique[k], np.mean(table_k["TPR"].dropna().to_numpy(), axis = 0), np.std(table_k["TPR"].dropna().to_numpy(), axis = 0, ddof = 1), np.mean(table_k["FPR"].dropna().to_numpy(), axis = 0), np.std(table_k["FPR"].dropna().to_numpy(), axis = 0, ddof = 1), np.mean(table_k["AUC"].dropna().to_numpy()), np.std(table_k["AUC"].dropna().to_numpy(), ddof = 1)])
    table_mean = pd.DataFrame(table_mean, columns=["input", "mean TPR", "std TPR", "mean FPR", "std FPR", "mean AUC", "std AUC"])

    # plot the ROC curve
    plt.figure(figsize = single_column_fig_size)
    plt.grid(alpha = 0.2)
    linestyles = ["-.", "--"]
    labels = ["CTA images", "Pattern spectra"]
    for i in range(len(args_input_unique)):
        table_mean_i = table_mean.copy()
        mean_true_positive_rate = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean TPR"].dropna().to_numpy()[0]
        std_true_positive_rate = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std TPR"].dropna().to_numpy()[0]
        mean_false_positive_rate = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean FPR"].dropna().to_numpy()[0]
        std_false_positive_rate = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std FPR"].dropna().to_numpy()[0]
        mean_AUC = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean AUC"].dropna().to_numpy()[0]
        std_AUC = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std AUC"].dropna().to_numpy()[0]

        plt.plot(mean_false_positive_rate, mean_true_positive_rate, linestyle = linestyles[i], color = colors_categorial[i], label = "{0} (AUC=${1:.3f}\pm{2:.3f}$)".format(labels[i], np.round(mean_AUC, 3), np.round(std_AUC, 3)))#"CTA images (AUC = {0:.3f})".format(np.round(area_under_ROC_curve_all[i], 3)))
        plt.fill_between(mean_false_positive_rate, mean_true_positive_rate - std_true_positive_rate, mean_true_positive_rate + std_true_positive_rate, facecolor = colors_categorial[i], alpha = 0.3)
        plt.fill_betweenx(mean_true_positive_rate, mean_false_positive_rate - std_false_positive_rate, mean_false_positive_rate + std_false_positive_rate, facecolor = colors_categorial[i], alpha = 0.3)

    plt.plot(np.linspace(0, 1, 5), np.linspace(0, 1, 5), color = "black", linestyle = "-")
    plt.xlabel(r"$\eta_{{p}}$")
    plt.ylabel(r"$\eta_{{\gamma}}$")
    plt.legend(loc = "lower right")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotAccuracyGammanessComparison(accuracy_gammaness_all, thresholds_all, input, path):
    # plot the ROC curve
    plt.figure(figsize = single_column_fig_size)
    plt.grid(alpha = 0.2)
    linestyles = ["-.", "--"]
    for i in range(len(accuracy_gammaness_all)):
        if input[i] == "cta":
            plt.plot(thresholds_all[i], accuracy_gammaness_all[i], linestyle = linestyles[0], color = colors_categorial[0], label = "CTA images") # \nCC $\gamma$ = {1:.3f}\nCC $p$ = {2:.3f}.format(np.round(area_under_ROC_curve, 3), np.round(true_positive_rate_50[0], 3), np.round(true_negative_rate_50[0], 3)))
        elif input[i] == "ps":
            plt.plot(thresholds_all[i], accuracy_gammaness_all[i], linestyle = linestyles[1], color = colors_categorial[1], label = "Pattern spectra") # \nCC $\gamma$ = {1:.3f}\nCC $p$ = {2:.3f}.format(np.round(area_under_ROC_curve, 3), np.round(true_positive_rate_50[0], 3), np.round(true_negative_rate_50[0], 3)))
    plt.xlabel("Gammaness")
    plt.ylabel("Accuracy")
    plt.legend(loc = "lower center")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotPrecisionGammanessComparison(precision_gammaness_all, thresholds_all, input, path):
    # plot the ROC curve
    plt.figure(figsize = single_column_fig_size)
    plt.grid(alpha = 0.2)
    linestyles = ["-.", "--"]
    for i in range(len(precision_gammaness_all)):
        if input[i] == "cta":
            plt.plot(thresholds_all[i], precision_gammaness_all[i], linestyle = linestyles[0], color = colors_categorial[0], label = "CTA images") # \nCC $\gamma$ = {1:.3f}\nCC $p$ = {2:.3f}.format(np.round(area_under_ROC_curve, 3), np.round(true_positive_rate_50[0], 3), np.round(true_negative_rate_50[0], 3)))
        elif input[i] == "ps":
            plt.plot(thresholds_all[i], precision_gammaness_all[i], linestyle = linestyles[1], color = colors_categorial[1], label = "Pattern spectra") # \nCC $\gamma$ = {1:.3f}\nCC $p$ = {2:.3f}.format(np.round(area_under_ROC_curve, 3), np.round(true_positive_rate_50[0], 3), np.round(true_negative_rate_50[0], 3)))
    plt.xlabel("Gammaness")
    plt.ylabel("Precision")
    plt.legend(loc = "lower right")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotPurityGammanessComparison(thresholds_all, true_positive_rate_all, false_positive_rate_all, args_input, path):
    table = []
    for k in range(len(args_input)):
        table.append([args_input[k], true_positive_rate_all[k], false_positive_rate_all[k]])

    table = pd.DataFrame(table, columns=["input", "TPR", "FPR"])

    table_mean = []
    args_input_unique = np.unique(args_input)
    for k in range(len(args_input_unique)):
        table_k = table.copy()
        table_k.where(table_k["input"] == args_input_unique[k], inplace = True)
        table_mean.append([args_input_unique[k], np.mean(table_k["TPR"].dropna().to_numpy(), axis = 0), np.std(table_k["TPR"].dropna().to_numpy(), axis = 0, ddof = 1), np.mean(table_k["FPR"].dropna().to_numpy(), axis = 0), np.std(table_k["FPR"].dropna().to_numpy(), axis = 0, ddof = 1)])
    table_mean = pd.DataFrame(table_mean, columns=["input", "mean TPR", "std TPR", "mean FPR", "std FPR"])

    plt.figure(figsize = single_column_fig_size_legend)
    plt.grid(alpha = 0.2)
    linestyles = ["-", "--"]
    labels = ["CTA images", "Pattern spectra"]
    for i in range(len(args_input_unique)):
        table_mean_i = table_mean.copy()
        mean_true_positive_rate = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean TPR"].dropna().to_numpy()[0]
        std_true_positive_rate = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std TPR"].dropna().to_numpy()[0]
        mean_false_positive_rate = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean FPR"].dropna().to_numpy()[0]
        std_false_positive_rate = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std FPR"].dropna().to_numpy()[0]

        plt.plot(thresholds_all[0], mean_true_positive_rate, linestyle = linestyles[0], color = colors_categorial[i], label = r"$\eta_{{\gamma}}$ ({0})".format(labels[i]))
        plt.fill_between(thresholds_all[0], mean_true_positive_rate - std_true_positive_rate, mean_true_positive_rate + std_true_positive_rate, facecolor = colors_categorial[i], alpha = 0.3)
        plt.plot(thresholds_all[0], mean_false_positive_rate, linestyle = linestyles[1], color = colors_categorial[i], label = r"$\eta_{{p}}$ ({0})".format(labels[i]))
        plt.fill_between(thresholds_all[0], mean_false_positive_rate - std_false_positive_rate, mean_false_positive_rate + std_false_positive_rate, facecolor = colors_categorial[i], alpha = 0.3)

    plt.xlabel(r"$\alpha_{g}$")
    plt.ylabel(r"$\eta$")
    plt.legend(bbox_to_anchor=(0., 1. , 1., .102), loc="lower left", mode = "expand", ncol = 2)
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()



    # plt.figure(figsize = single_column_fig_size_legend)
    # plt.grid(alpha = 0.2)
    # linestyles = ["-.", "--"]

    # for i in range(len(thresholds_all)):
    #     if input[i] == "cta":
    #         plt.plot(thresholds_all[i], true_positive_rate_all[i], linestyle = "solid", color = colors_categorial[0], label = r"$\eta_{{\gamma}}$ (CTA images)", alpha = 1.0) 
    #         plt.plot(thresholds_all[i], false_positive_rate_all[i], linestyle = "dashed", color = colors_categorial[0], label = r"$\eta_{{p}}$ (CTA images)", alpha = 1.0)

    #         print("CTA")
    #         index_purity_10p = np.where(false_positive_rate_all[i] <= 1e-4)
    #         print("index_purity_10p", index_purity_10p)
    #         index_purity_10p = np.where(false_positive_rate_all[i] <= 1e-4)[0][0]
    #         print("index_purity_10p", index_purity_10p)
    #         threshold_purity_10p = thresholds_all[i][index_purity_10p]
    #         true_positive_rate_purity_10p = true_positive_rate_all[i][index_purity_10p]
    #         false_positive_rate_purity_10p = false_positive_rate_all[i][index_purity_10p]

    #         print("threshold_purity_10p", threshold_purity_10p)
    #         print("true_positive_rate_purity_10p", true_positive_rate_purity_10p)
    #         print("false_positive_rate_purity_10p", false_positive_rate_purity_10p)
    #         print("______________________")

    #     elif input[i] == "ps":
    #         plt.plot(thresholds_all[i], true_positive_rate_all[i], linestyle = "solid", color = colors_categorial[1], label = r"$\eta_{{\gamma}}$ (pattern spectra)", alpha = 1.0) 
    #         plt.plot(thresholds_all[i], false_positive_rate_all[i], linestyle = "dashed", color = colors_categorial[1], label = r"$\eta_{{p}}$ (pattern spectra)", alpha = 1.0) 

    #         print("PS")
    #         index_purity_10p = np.where(false_positive_rate_all[i] <= 1e-4)
    #         print("index_purity_10p", index_purity_10p)
    #         index_purity_10p = np.where(false_positive_rate_all[i] <= 1e-4)[0][0]
    #         print("index_purity_10p", index_purity_10p)
    #         threshold_purity_10p = thresholds_all[i][index_purity_10p]
    #         true_positive_rate_purity_10p = true_positive_rate_all[i][index_purity_10p]
    #         false_positive_rate_purity_10p = false_positive_rate_all[i][index_purity_10p]

    #         print("threshold_purity_10p", threshold_purity_10p)
    #         print("true_positive_rate_purity_10p", true_positive_rate_purity_10p)
    #         print("false_positive_rate_purity_10p", false_positive_rate_purity_10p)
    #         print("______________________")

    # plt.xlabel(r"$\alpha_{g}$")
    # plt.ylabel(r"$\eta$")
    # plt.legend(bbox_to_anchor=(0., 1. , 1., .102), loc="lower left", mode = "expand", ncol = 2)
    # plt.tight_layout()
    # plt.savefig(path, dpi = 250)
    # plt.close()

def PlotAUCEnergyComparison(bins, bins_central, area_under_ROC_curve_energy_all, args_input, path):
    table = []
    for k in range(len(args_input)):
        table.append([args_input[k], area_under_ROC_curve_energy_all[k]])

    table = pd.DataFrame(table, columns=["input", "AUC"])

    table_mean = []
    args_input_unique = np.unique(args_input)
    for k in range(len(args_input_unique)):
        table_k = table.copy()
        table_k.where(table_k["input"] == args_input_unique[k], inplace = True)
        table_mean.append([args_input_unique[k], np.mean(table_k["AUC"].dropna().to_numpy(), axis = 0), np.std(table_k["AUC"].dropna().to_numpy(), axis = 0, ddof = 1)])
    table_mean = pd.DataFrame(table_mean, columns=["input", "mean AUC", "std AUC"])

    plt.figure(figsize = single_column_fig_size)
    plt.grid(alpha = 0.2)
    labels = ["CTA images", "Pattern spectra"]
    for i in range(len(args_input_unique)):
        table_mean_i = table_mean.copy()
        mean_AUC = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean AUC"].dropna().to_numpy()[0]
        std_AUC = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std AUC"].dropna().to_numpy()[0]
        plt.errorbar(bins_central, (1 - mean_AUC), xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 0.0, marker = markers[i], markersize = markersizes[i], label = labels[i], color = colors_categorial[i])
        bins_central_fill = np.append(bins_central, bins[-1])
        bins_central_fill = np.insert(bins_central_fill, 0, bins[0])
        filling_lower = (1 - mean_AUC) - std_AUC
        filling_lower = np.append(filling_lower, filling_lower[-1])
        filling_lower = np.insert(filling_lower, 0, filling_lower[0])
        filling_upper = (1 - mean_AUC) + std_AUC
        filling_upper = np.append(filling_upper, filling_upper[-1])
        filling_upper = np.insert(filling_upper, 0, filling_upper[0])
        plt.fill_between(bins_central_fill, filling_lower, filling_upper, facecolor = colors_categorial[i], alpha = 0.3)
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
    plt.ylabel("1 - AUC")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotAccuracyEnergyComparison(bins, bins_central, accuracy_energy_all, args_input, path):
    table = []
    for k in range(len(args_input)):
        table.append([args_input[k], accuracy_energy_all[k]])

    table = pd.DataFrame(table, columns=["input", "accuraccy"])

    table_mean = []
    args_input_unique = np.unique(args_input)
    for k in range(len(args_input_unique)):
        table_k = table.copy()
        table_k.where(table_k["input"] == args_input_unique[k], inplace = True)
        table_mean.append([args_input_unique[k], np.mean(table_k["accuraccy"].dropna().to_numpy(), axis = 0), np.std(table_k["accuraccy"].dropna().to_numpy(), axis = 0, ddof = 1)])
    table_mean = pd.DataFrame(table_mean, columns=["input", "mean accuracy", "std accuracy"])

    plt.figure(figsize = single_column_fig_size)
    plt.grid(alpha = 0.2)
    labels = ["CTA images", "Pattern spectra"]
    for i in range(len(args_input_unique)):
        table_mean_i = table_mean.copy()
        mean_accuracy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean accuracy"].dropna().to_numpy()[0]
        std_accuracy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std accuracy"].dropna().to_numpy()[0]
        plt.errorbar(bins_central, (1 - mean_accuracy), xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 0.0, marker = markers[i], markersize = markersizes[i], label = labels[i], color = colors_categorial[i])
        bins_central_fill = np.append(bins_central, bins[-1])
        bins_central_fill = np.insert(bins_central_fill, 0, bins[0])
        filling_lower = (1 - mean_accuracy) - std_accuracy
        filling_lower = np.append(filling_lower, filling_lower[-1])
        filling_lower = np.insert(filling_lower, 0, filling_lower[0])
        filling_upper = (1 - mean_accuracy) + std_accuracy
        filling_upper = np.append(filling_upper, filling_upper[-1])
        filling_upper = np.insert(filling_upper, 0, filling_upper[0])
        plt.fill_between(bins_central_fill, filling_lower, filling_upper, facecolor = colors_categorial[i], alpha = 0.3)
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
    plt.ylabel("1 - accuracy")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

# def PlotEfficiencyEnergyComparison(bins, bins_central, true_positive_rate_energy_all, false_positive_rate_energy_all, args_input, path):
#     table = []
#     for k in range(len(args_input)):
#         table.append([args_input[k], true_positive_rate_energy_all[k], false_positive_rate_energy_all[k]])

#     table = pd.DataFrame(table, columns=["input", "TPR energy", "FPR energy"])

#     table_mean = []
#     args_input_unique = np.unique(args_input)
#     for k in range(len(args_input_unique)):
#         table_k = table.copy()
#         table_k.where(table_k["input"] == args_input_unique[k], inplace = True)
#         table_mean.append([args_input_unique[k], np.mean(table_k["TPR energy"].dropna().to_numpy(), axis = 0), np.std(table_k["TPR energy"].dropna().to_numpy(), axis = 0, ddof = 1), np.mean(table_k["FPR energy"].dropna().to_numpy(), axis = 0), np.std(table_k["FPR energy"].dropna().to_numpy(), axis = 0, ddof = 1)])
#     table_mean = pd.DataFrame(table_mean, columns=["input", "mean TPR energy", "std TPR energy", "mean FPR energy", "std FPR energy"])

#     print(table_mean)

#     plt.figure(figsize = single_column_fig_size_legend)
#     plt.grid(alpha = 0.2)
#     labels = ["CTA images", "pattern spectra"]
#     for i in range(len(args_input_unique)):
#         table_mean_i = table_mean.copy()
#         mean_true_positive_rate_energy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean TPR energy"].dropna().to_numpy()[0]
#         std_true_positive_rate_energy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std TPR energy"].dropna().to_numpy()[0]
#         mean_false_positive_rate_energy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean FPR energy"].dropna().to_numpy()[0]
#         std_false_positive_rate_energy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std FPR energy"].dropna().to_numpy()[0]
#         plt.errorbar(bins_central, (mean_true_positive_rate_energy), xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 0.0, marker = ".", markersize = 6, label = r"$\eta_{{\gamma}}$ ({0})".format(labels[i]), color = colors_categorial[i])
#         plt.errorbar(bins_central, (mean_false_positive_rate_energy), xerr = (bins[:-1] - bins_central, bins_central - bins[1:]), linestyle = "", capsize = 0.0, marker = "s", markersize = 3, label = r"$\eta_{{p}}$ ({0})".format(labels[i]), color = colors_categorial[i], fillstyle = "none")
#         bins_central_fill = np.append(bins_central, bins[-1])
#         bins_central_fill = np.insert(bins_central_fill, 0, bins[0])

#         filling_lower_tpr = (mean_true_positive_rate_energy) - std_true_positive_rate_energy
#         filling_lower_tpr = np.append(filling_lower_tpr, filling_lower_tpr[-1])
#         filling_lower_tpr = np.insert(filling_lower_tpr, 0, filling_lower_tpr[0])
#         filling_upper_tpr = (mean_true_positive_rate_energy) + std_true_positive_rate_energy
#         filling_upper_tpr = np.append(filling_upper_tpr, filling_upper_tpr[-1])
#         filling_upper_tpr = np.insert(filling_upper_tpr, 0, filling_upper_tpr[0])
        
#         filling_lower_fpr = (mean_false_positive_rate_energy) - std_false_positive_rate_energy
#         filling_lower_fpr = np.append(filling_lower_fpr, filling_lower_fpr[-1])
#         filling_lower_fpr = np.insert(filling_lower_fpr, 0, filling_lower_fpr[0])
#         filling_upper_fpr = (mean_false_positive_rate_energy) + std_false_positive_rate_energy
#         filling_upper_fpr = np.append(filling_upper_fpr, filling_upper_fpr[-1])
#         filling_upper_fpr = np.insert(filling_upper_fpr, 0, filling_upper_fpr[0])

#         plt.fill_between(bins_central_fill, filling_lower_tpr, filling_upper_tpr, facecolor = colors_categorial[i], alpha = 0.3)
#         plt.fill_between(bins_central_fill, filling_lower_fpr, filling_upper_fpr, facecolor = colors_categorial[i], alpha = 0.3)
#     plt.xlabel("$E_\mathrm{true}$ [TeV]")
#     plt.ylabel(r"$\eta$")
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.legend(bbox_to_anchor=(0., 1. , 1., .102), loc="lower left", mode = "expand", ncol = 2)
#     plt.tight_layout()
#     plt.savefig(path, dpi = 250)
#     plt.close()

def PlotEfficiencyEnergyComparison(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, true_positive_rate_er_gamma, false_positive_rate_er_proton, args_input, path):

    table = []
    for k in range(len(args_input)):
        table.append([args_input[k], true_positive_rate_er_gamma[k], false_positive_rate_er_proton[k]])

    table = pd.DataFrame(table, columns=["input", "TPR energy", "FPR energy"])

    table_mean = []
    args_input_unique = np.unique(args_input)
    for k in range(len(args_input_unique)):
        table_k = table.copy()
        table_k.where(table_k["input"] == args_input_unique[k], inplace = True)
        table_mean.append([args_input_unique[k], np.mean(table_k["TPR energy"].dropna().to_numpy(), axis = 0), np.std(table_k["TPR energy"].dropna().to_numpy(), axis = 0, ddof = 1), np.mean(table_k["FPR energy"].dropna().to_numpy(), axis = 0), np.std(table_k["FPR energy"].dropna().to_numpy(), axis = 0, ddof = 1)])
    table_mean = pd.DataFrame(table_mean, columns=["input", "mean TPR energy", "std TPR energy", "mean FPR energy", "std FPR energy"])

    plt.figure(figsize = single_column_fig_size_legend)
    plt.grid(alpha = 0.2)
    labels = ["CTA images", "pattern spectra"]
    for i in range(len(args_input_unique)):
        table_mean_i = table_mean.copy()
        mean_true_positive_rate_energy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean TPR energy"].dropna().to_numpy()[0]
        std_true_positive_rate_energy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std TPR energy"].dropna().to_numpy()[0]
        mean_false_positive_rate_energy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean FPR energy"].dropna().to_numpy()[0]
        std_false_positive_rate_energy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std FPR energy"].dropna().to_numpy()[0]
        plt.errorbar(bins_central_gamma, (mean_true_positive_rate_energy), xerr = (bins_gamma[:-1] - bins_central_gamma, bins_central_gamma - bins_gamma[1:]), linestyle = "", capsize = 0.0, marker = ".", markersize = 6, label = r"$\eta_{{\gamma}}$ ({0})".format(labels[i]), color = colors_categorial[i])
        plt.errorbar(bins_central_proton, (mean_false_positive_rate_energy), xerr = (bins_proton[:-1] - bins_central_proton, bins_central_proton - bins_proton[1:]), linestyle = "", capsize = 0.0, marker = "s", markersize = 3, label = r"$\eta_{{p}}$ ({0})".format(labels[i]), color = colors_categorial[i], fillstyle = "none")
        bins_central_fill_gamma = np.append(bins_central_gamma, bins_gamma[-1])
        bins_central_fill_proton = np.append(bins_central_proton, bins_proton[-1])
        bins_central_fill_gamma = np.insert(bins_central_fill_gamma, 0, bins_gamma[0])
        bins_central_fill_proton = np.insert(bins_central_fill_proton, 0, bins_proton[0])

        filling_lower_tpr = (mean_true_positive_rate_energy) - std_true_positive_rate_energy
        filling_lower_tpr = np.append(filling_lower_tpr, filling_lower_tpr[-1])
        filling_lower_tpr = np.insert(filling_lower_tpr, 0, filling_lower_tpr[0])
        filling_upper_tpr = (mean_true_positive_rate_energy) + std_true_positive_rate_energy
        filling_upper_tpr = np.append(filling_upper_tpr, filling_upper_tpr[-1])
        filling_upper_tpr = np.insert(filling_upper_tpr, 0, filling_upper_tpr[0])
        
        filling_lower_fpr = (mean_false_positive_rate_energy) - std_false_positive_rate_energy
        filling_lower_fpr = np.append(filling_lower_fpr, filling_lower_fpr[-1])
        filling_lower_fpr = np.insert(filling_lower_fpr, 0, filling_lower_fpr[0])
        filling_upper_fpr = (mean_false_positive_rate_energy) + std_false_positive_rate_energy
        filling_upper_fpr = np.append(filling_upper_fpr, filling_upper_fpr[-1])
        filling_upper_fpr = np.insert(filling_upper_fpr, 0, filling_upper_fpr[0])

        plt.fill_between(bins_central_fill_gamma, filling_lower_tpr, filling_upper_tpr, facecolor = colors_categorial[i], alpha = 0.3)
        plt.fill_between(bins_central_fill_proton, filling_lower_fpr, filling_upper_fpr, facecolor = colors_categorial[i], alpha = 0.3)
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
    plt.ylabel(r"$\eta$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(0., 1. , 1., .102), loc="lower left", mode = "expand", ncol = 2)
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotGammaEfficiencyEnergyComparison(bins_gamma, bins_proton, bins_central_gamma, bins_central_proton, true_positive_rate_er_gamma, false_positive_rate_er_proton, args_input, path):

    table = []
    for k in range(len(args_input)):
        table.append([args_input[k], true_positive_rate_er_gamma[k], false_positive_rate_er_proton[k]])

    table = pd.DataFrame(table, columns=["input", "TPR energy", "FPR energy"])

    table_mean = []
    args_input_unique = np.unique(args_input)
    for k in range(len(args_input_unique)):
        table_k = table.copy()
        table_k.where(table_k["input"] == args_input_unique[k], inplace = True)
        table_mean.append([args_input_unique[k], np.mean(table_k["TPR energy"].dropna().to_numpy(), axis = 0), np.std(table_k["TPR energy"].dropna().to_numpy(), axis = 0, ddof = 1), np.mean(table_k["FPR energy"].dropna().to_numpy(), axis = 0), np.std(table_k["FPR energy"].dropna().to_numpy(), axis = 0, ddof = 1)])
    table_mean = pd.DataFrame(table_mean, columns=["input", "mean TPR energy", "std TPR energy", "mean FPR energy", "std FPR energy"])

    plt.figure(figsize = single_column_fig_size_legend)
    plt.grid(alpha = 0.2)
    labels = ["CTA images", "Pattern spectra"]
    for i in range(len(args_input_unique)):
        table_mean_i = table_mean.copy()
        mean_true_positive_rate_energy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["mean TPR energy"].dropna().to_numpy()[0]
        std_true_positive_rate_energy = table_mean_i.where(table_mean_i["input"] == args_input_unique[i])["std TPR energy"].dropna().to_numpy()[0]
        plt.errorbar(bins_central_gamma, (mean_true_positive_rate_energy), xerr = (bins_gamma[:-1] - bins_central_gamma, bins_central_gamma - bins_gamma[1:]), linestyle = "", capsize = 0.0, marker = ".", markersize = 6, label = r"{0}".format(labels[i]), color = colors_categorial[i])
        bins_central_fill_gamma = np.append(bins_central_gamma, bins_gamma[-1])
        bins_central_fill_gamma = np.insert(bins_central_fill_gamma, 0, bins_gamma[0])

        filling_lower_tpr = (mean_true_positive_rate_energy) - std_true_positive_rate_energy
        filling_lower_tpr = np.append(filling_lower_tpr, filling_lower_tpr[-1])
        filling_lower_tpr = np.insert(filling_lower_tpr, 0, filling_lower_tpr[0])
        filling_upper_tpr = (mean_true_positive_rate_energy) + std_true_positive_rate_energy
        filling_upper_tpr = np.append(filling_upper_tpr, filling_upper_tpr[-1])
        filling_upper_tpr = np.insert(filling_upper_tpr, 0, filling_upper_tpr[0])
        
        plt.fill_between(bins_central_fill_gamma, filling_lower_tpr, filling_upper_tpr, facecolor = colors_categorial[i], alpha = 0.3)
        # plt.fill_between(bins_central_fill_proton, filling_lower_fpr, filling_upper_fpr, facecolor = colors_categorial[i], alpha = 0.3)
    plt.xlabel("$E_\mathrm{true}$ [TeV]")
    plt.ylabel(r"$\eta_{{\gamma}}$")
    plt.xscale("log")
    plt.yscale("log")
    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim(ymin, ymax * 1.3)
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.savefig(path, dpi = 250)
    plt.close()

def MeanStdAUC(area_under_ROC_curve_all, input):
    area_under_ROC_curve_all_cta = np.array([])
    area_under_ROC_curve_all_ps = np.array([])
    for i in range(len(input)):
        if input[i] == "cta":
            area_under_ROC_curve_all_cta = np.append(area_under_ROC_curve_all_cta, area_under_ROC_curve_all[i])
        elif input[i] == "ps":
            area_under_ROC_curve_all_ps = np.append(area_under_ROC_curve_all_ps, area_under_ROC_curve_all[i])

    area_under_ROC_curve_all_cta_mean, area_under_ROC_curve_all_cta_std = np.mean(area_under_ROC_curve_all_cta), np.std(area_under_ROC_curve_all_cta, ddof = 1)
    area_under_ROC_curve_all_ps_mean, area_under_ROC_curve_all_ps_std = np.mean(area_under_ROC_curve_all_ps), np.std(area_under_ROC_curve_all_ps, ddof = 1)

    # print("CTA images AUC value: {0:.3f} +- {1:.3f}".format(area_under_ROC_curve_all_cta_mean, area_under_ROC_curve_all_cta_std))
    # print("Pattern spectra AUC value: {0:.3f} +- {1:.3f}".format(area_under_ROC_curve_all_ps_mean, area_under_ROC_curve_all_ps_std))



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
            # fig.suptitle(f"{particle_type[pt]} examples - {gammaness_limit[0]} < gammaness < {gammaness_limit[1]}")
            path_total = path + "_" + particle_type[pt] + f"gl_{gammaness_limit[0]}_{gammaness_limit[1]}" + ".png"
        elif particle_type[pt] == "proton":
            # fig.suptitle(f"{particle_type[pt]} examples - {gammaness_limit[2]} < gammaness < {gammaness_limit[3]}")
            path_total = path + "_" + particle_type[pt] + f"gl_{gammaness_limit[2]}_{gammaness_limit[3]}" + ".png"
        ax = ax.ravel()
        for n in range(N**2):
            # ax[n].title.set_text(f"{int(np.round(table_gammaness_limit['E_true / GeV'][n]))} GeV")
            ax[n].imshow(table_gammaness_limit[string_table_column][n], cmap = "Greys_r")
            ax[n].set_xticks([])
            ax[n].set_yticks([])
        # plt.tight_layout()
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
        #     plt.title(f"{particle_type[pt]} mean - {gammaness_limit[0]} < gammaness < {gammaness_limit[1]}", fontsize = 12)
            path_total = path + "_" + particle_type[pt] + f"_gl_{gammaness_limit[0]}_{gammaness_limit[1]}" + ".png"
        elif particle_type[pt] == "proton":
        #     plt.title(f"{particle_type[pt]} mean - {gammaness_limit[2]} < gammaness < {gammaness_limit[3]}", fontsize = 12)
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
    # plt.title(f"pattern spectra mean difference - {particle_type[0]} - {particle_type[1]}" "\n" f"{gammaness_limit[0]} ({gammaness_limit[2]}) < gammaness < {gammaness_limit[1]} ({gammaness_limit[3]})", fontsize = 10)
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

def SaveCSV(y, bins, y_label, path):
    table = pd.DataFrame()
    table["E_min [TeV]"] = bins[:-1]
    table["E_max [TeV]"] = bins[1:]
    table[f"{y_label}"] = y
    table.to_csv(path)
