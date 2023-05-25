import matplotlib.pyplot as plt
from keras.layers import Add, Conv2D, ReLU
import numpy as np
import pandas as pd

def ResBlock(z, kernelsizes, filters, increase_dim = False):
    """
    Residual block for ResNet
    Args:
        z: input tensor
        kernelsizes: list of kernelsizes for the three convolutional layers
        filters: list of filters for the three convolutional layers
        increase_dim: boolean to increase the dimension of the input tensor
    Returns:
        out: output tensor
    
    """

    z_shortcut = z
    kernelsize_1, kernelsize_2 = kernelsizes
    filters_1, filters_2 = filters

    fz = Conv2D(filters_1, kernelsize_1)(z)
    fz = ReLU()(fz)

    fz = Conv2D(filters_1, kernelsize_2, padding = "same")(fz)
    fz = ReLU()(fz)
    
    fz = Conv2D(filters_2, kernelsize_1)(fz)

    if increase_dim == True:
        z_shortcut = Conv2D(filters_2, (1, 1))(z_shortcut)

    out = Add()([fz, z_shortcut])
    out = ReLU()(out)
    
    return out

def PlotExamplesEnergy(X, Y, path):
    """
    Plot a few examples of the input data
    Args:
        X: input data
        Y: true energy
        path: path to save the plot
    """
    # plot a few examples
    fig, ax = plt.subplots(3, 3)
    ax = ax.ravel()
    for i in range(9):
        ax[i].imshow(X[i], cmap = "Greys_r")
        ax[i].title.set_text(f"{int(np.round(10**Y[i]))} GeV")
        ax[i].axis("off")
    plt.savefig(path, dpi = 250)
    plt.close()

def PlotEnergyDistributionEnergy(Y, path):
    """
    Plot the energy distribution
    Args:
        Y: true energy
        path: path to save the plot
    """
    plt.figure()
    plt.hist(10**Y, bins=np.logspace(np.log10(np.min(10**Y)),np.log10(np.max(10**Y)), 50))
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Number of events")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(path, dpi = 250)
    plt.close()

def GetDataEnergyReconstruction(small_dataset, string_input, string_ps_input, string_input_short, energy_range_gamma, selection_cuts_train, selection_cuts_test, telescope_mode, split_percentage):
    """
    Load the data for the energy reconstruction
    Args:
        small_dataset: string to indicate if the small dataset is used
        string_input: string to indicate the input data
        string_ps_input: string to indicate the input data for pattern spectra
        string_input_short: string to indicate the input data (short)
        energy_range_gamma: energy range for gammas
        selection_cuts_train: string to indicate the selection cuts for training
        selection_cuts_test: string to indicate the selection cuts for testing
        telescope_mode: string to indicate the telescope mode
        split_percentage: percentage to split the data into training and test data
    Returns:
        table_train: training data
        table_test: test data
        """
    # load csv file with list of runs
    if small_dataset == "y":
        filename_run_csv = f"scripts/run_lists/gamma_run_list_alpha_test.csv"
    elif telescope_mode == "mono":
        filename_run_csv = f"scripts/run_lists/gamma_run_list_mono_alpha.csv"
    else: 
        filename_run_csv = f"scripts/run_lists/gamma_run_list_alpha.csv"
    run = pd.read_csv(filename_run_csv)
    run = run.to_numpy().reshape(len(run))

    # start counter for number of events
    events_count = 0
    events_count_energy_cut = 0

    # initialise train and test table
    table_train = pd.DataFrame()
    table_test = pd.DataFrame()

    # loop over each run
    for r in range(len(run)):
        # load data from run
        run_filename = f"gamma_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
        input_filename = f"cnn/{string_input}/input/gamma/" + string_ps_input + run_filename + string_input_short + ".h5"

        table_run = pd.read_hdf(input_filename)
        print(f"Number of events in Run {run[r]}:", len(table_run))
        events_count += len(table_run)

        # apply energy cut
        table_run.drop(table_run.loc[table_run["true_energy"] <= energy_range_gamma[0] * 1e3].index, inplace=True)
        table_run.drop(table_run.loc[table_run["true_energy"] >= energy_range_gamma[1] * 1e3].index, inplace=True)
        table_run.reset_index(inplace = True)
        events_count_energy_cut += len(table_run)

        # shuffle data set
        table_run = table_run.sample(frac = 1).reset_index(drop = True)

        # split data into training and test data
        table_run_test, table_run_train = np.split(table_run, [int(len(table_run) * split_percentage)])

        # apply selection cut on training data
        if selection_cuts_train != None:
            selection_cuts_train_filename = f"cnn/selection_cuts/gamma/{selection_cuts_train}/run{run[r]}.csv"
            table_selection_cuts_train = pd.read_csv(selection_cuts_train_filename)
            if telescope_mode == "stereo_sum_cta":
                merged_table_train = pd.merge(table_run_train, table_selection_cuts_train, on=["obs_id", "event_id"])
            elif telescope_mode == "mono":
                merged_table_train = pd.merge(table_run_train, table_selection_cuts_train, on=["obs_id", "event_id", "tel_id"])
        
            table_train = table_train.append(merged_table_train, ignore_index = True)
        else:
            table_train = table_train.append(table_run_train, ignore_index = True)

        # apply selection cut on test data
        if selection_cuts_test != None:
            selection_cuts_test_filename = f"cnn/selection_cuts/gamma/{selection_cuts_test}/run{run[r]}.csv"
            table_selection_cuts_test = pd.read_csv(selection_cuts_test_filename)
            if telescope_mode == "stereo_sum_cta":
                merged_table_test = pd.merge(table_run_test, table_selection_cuts_test, on=["obs_id", "event_id"])
            elif telescope_mode == "mono":
                merged_table_test = pd.merge(table_run_test, table_selection_cuts_test, on=["obs_id", "event_id", "tel_id"])
        
            table_test = table_test.append(merged_table_test, ignore_index = True)
        else:
            table_test = table_test.append(table_run_test, ignore_index = True)
    
    print("Total number of events:", events_count)
    print("Total number of events after energy cut:", events_count_energy_cut)
    print("______________________________________________")
    if selection_cuts_train != None:
        print("Total number of training events after selection cut:", len(table_train))
    if selection_cuts_test != None:
        print("Total number of testing events after selection cut:", len(table_test))
    if selection_cuts_train != None or selection_cuts_test != None:
        print("Total number of all events after selection cut:", len(table_train) + len(table_test))
        print("______________________________________________")

    return(table_train, table_test)

def GetDataSeparation(small_dataset, telescope_mode, string_input, string_ps_input, string_input_short, energy_range_gamma, energy_range_proton, selection_cuts_train, selection_cuts_test, split_percentage):
    """
    Load the data for the particle separation
    Args:
        small_dataset: string to indicate if the small dataset is used
        telescope_mode: string to indicate the telescope mode
        string_input: string to indicate the input data
        string_ps_input: string to indicate the input data for pattern spectra
        string_input_short: string to indicate the input data (short)
        energy_range_gamma: energy range for gammas
        energy_range_proton: energy range for protons
        selection_cuts_train: string to indicate the selection cuts for training
        selection_cuts_test: string to indicate the selection cuts for testing
        split_percentage: percentage to split the data into training and test data
    Returns:
        table_train: training data
        table_test: test data
        """
    particle_type = np.array(["gamma_diffuse", "proton"])

    table_train = pd.DataFrame()
    table_test = pd.DataFrame()

    events_count = np.array([0, 0])
    events_count_energy_cut = np.array([0, 0])
    events_count_selection_cuts_train = np.array([0, 0])
    events_count_selection_cuts_test = np.array([0, 0])
    for p in range(len(particle_type)):
        if small_dataset == "y":
            filename_run_csv = f"scripts/run_lists/{particle_type[p]}_run_list_alpha_test.csv"
        elif telescope_mode == "mono":
            filename_run_csv = f"scripts/run_lists/{particle_type[p]}_run_list_mono_alpha.csv"
        else: 
            filename_run_csv = f"scripts/run_lists/{particle_type[p]}_run_list_alpha.csv"
        run = pd.read_csv(filename_run_csv)
        run = run.to_numpy().reshape(len(run))

        print(f"List of runs {particle_type[p]} filename:")
        print(filename_run_csv)

        for r in range(len(run)):
            run_filename = f"{particle_type[p]}_20deg_0deg_run{run[r]}___cta-prod5-paranal_desert-2147m-Paranal-dark_merged.DL1"
            input_filename = f"cnn/{string_input}/input/{particle_type[p]}/" + string_ps_input + run_filename + string_input_short + ".h5"

            table_run = pd.read_hdf(input_filename)
            print(f"Number of events in {particle_type[p]} Run {run[r]}:", len(table_run))

            if (particle_type[p] == "gamma_diffuse"):
                events_count[0] += len(table_run)

                # apply energy cut for gammas
                table_run.drop(table_run.loc[(table_run["true_energy"] <= energy_range_gamma[0] * 1e3) & (table_run["particle"] == 1)].index, inplace=True)
                table_run.drop(table_run.loc[(table_run["true_energy"] >= energy_range_gamma[1] * 1e3) & (table_run["particle"] == 1)].index, inplace=True)

                events_count_energy_cut[0] += len(table_run)

            if particle_type[p] == "proton":
                events_count[1] += len(table_run)

                # apply energy cut for protons
                table_run.drop(table_run.loc[(table_run["true_energy"] <= energy_range_proton[0] * 1e3) & (table_run["particle"] == 0)].index, inplace=True)
                table_run.drop(table_run.loc[(table_run["true_energy"] >= energy_range_proton[1] * 1e3) & (table_run["particle"] == 0)].index, inplace=True)

                events_count_energy_cut[1] += len(table_run)

            # shuffle data set
            table_run = table_run.sample(frac = 1).reset_index(drop = True)

             # split data into training and test data
            table_run_test, table_run_train = np.split(table_run, [int(len(table_run) * split_percentage)])

            # apply selection cuts on training data
            if selection_cuts_train != None:
                selection_cuts_train_filename = f"cnn/selection_cuts/{particle_type[p]}/{selection_cuts_train}/run{run[r]}.csv"
                table_selection_cuts_train = pd.read_csv(selection_cuts_train_filename)
                if telescope_mode == "stereo_sum_cta":
                    merged_table_train = pd.merge(table_run_train, table_selection_cuts_train, on=["obs_id", "event_id"])
                elif telescope_mode == "mono":
                    merged_table_train = pd.merge(table_run_train, table_selection_cuts_train, on=["obs_id", "event_id", "tel_id"])

                if (particle_type[p] == "gamma_diffuse"):
                    events_count_selection_cuts_train[0] += len(merged_table_train)
                if particle_type[p] == "proton":
                    events_count_selection_cuts_train[1] += len(merged_table_train)
            
                table_train = table_train.append(merged_table_train, ignore_index = True)
            else:
                table_train = table_train.append(table_run_train, ignore_index = True)


            # apply selection cuts on test data
            if selection_cuts_test != None:
                selection_cuts_test_filename = f"cnn/selection_cuts/{particle_type[p]}/{selection_cuts_test}/run{run[r]}.csv"
                table_selection_cuts_test = pd.read_csv(selection_cuts_test_filename)
                if telescope_mode == "stereo_sum_cta":
                    merged_table_test = pd.merge(table_run_test, table_selection_cuts_test, on=["obs_id", "event_id"])
                elif telescope_mode == "mono":
                    merged_table_test = pd.merge(table_run_test, table_selection_cuts_test, on=["obs_id", "event_id", "tel_id"])

                if (particle_type[p] == "gamma_diffuse"):
                    events_count_selection_cuts_test[0] += len(merged_table_test)
                if particle_type[p] == "proton":
                    events_count_selection_cuts_test[1] += len(merged_table_test)
            
                table_test = table_test.append(merged_table_test, ignore_index = True)
            else:
                table_test = table_test.append(table_run_test, ignore_index = True)

            table_train = table_train.sample(frac = 1).reset_index(drop = True)

    print("______________________________________________")
    print("Total number of gamma events:", events_count[0])
    print("Total number of proton events:", events_count[1])

    print("______________________________________________")
    print("Total number of gamma events after energy cut:", events_count_energy_cut[0])
    print("Total number of proton events after energy cut:", events_count_energy_cut[1])


    if selection_cuts_train != None:
        print("______________________________________________")
        print("Total number of gamma training events after selection cuts:", events_count_selection_cuts_train[0])
        print("Total number of proton training events after selection cuts:", events_count_selection_cuts_train[1])
    if selection_cuts_test != None:
        print("______________________________________________")
        print("Total number of gamma test events after selection cuts:", events_count_selection_cuts_test[0])
        print("Total number of proton test events after selection cuts:", events_count_selection_cuts_test[1])
    if selection_cuts_train != None or selection_cuts_test != None:
        print("______________________________________________")
        print("Total number of training events after selection cuts:", len(table_train))
        print("Total number of test events after selection cuts:", len(table_test))
        print("Total number of all events after selection cuts:", len(table_train) +  len(table_test))

    return(table_train, table_test)

def GetUserInputStr(name, input, mode, telescope_mode, energy_range_gamma, energy_range_proton, selection_cuts_train, selection_cuts_test, epochs, small_dataset, attribute, domain_lower, domain_higher, mapper, size, filter):
    """
    Convert user input to relevant strings and prints a user input summary
    Args:
        name: name of the experiment
        input: string to indicate the input data
        mode: string to indicate the mode
        telescope_mode: string to indicate the telescope mode
        energy_range_gamma: energy range for gammas
        energy_range_proton: energy range for protons
        selection_cuts_train: string to indicate the selection cuts for training
        selection_cuts_test: string to indicate the selection cuts for testing
        epochs: number of epochs
        small_dataset: string to indicate if the small dataset is used
        attribute: attribute for pattern spectra
        domain_lower: lower domain for pattern spectra
        domain_higher: higher domain for pattern spectra
        mapper: mapper for pattern spectra
        size: size for pattern spectra
        filter: filter for pattern spectra
    Returns:
        string_name: string to indicate the name of the experiment
        string_input: string to indicate the input data
        string_ps_input: string to indicate the input data for pattern spectra
        string_input_short: string to indicate the input data (short)
        string_table_column: string to indicate the column name in the table
        """
    if name != None:
        string_name = f"_{name}"
    else:
        string_name = ""

    if input == "cta":
        print(f"################### Input summary ################### \nMode: {mode} \nInput: CTA images \nEnergy range (gamma/proton): {energy_range_gamma} / {energy_range_proton} TeV \nSelection cuts: {selection_cuts_train} (training), {selection_cuts_test} (testing) \nEpochs: {epochs} \nTest run: {small_dataset}")
        string_input = "iact_images"
        if telescope_mode == "stereo_sum_cta":
            string_input_short = "_images_alpha"
        elif telescope_mode == "mono":
            string_input_short = "_images_mono_alpha"
        string_ps_input = ""
        string_table_column = "image"
    elif input == "ps":
        print(f"################### Input summary ################### \nMode: {mode} \nInput: pattern spectra \nEnergy range (gamma/proton): {energy_range_gamma} / {energy_range_proton} TeV \nSelection cuts: {selection_cuts_train} (training), {selection_cuts_test} (testing) \nAttribute: {attribute} \nDomain lower: {domain_lower} \nDomain higher: {domain_higher} \nMapper: {mapper} \nSize: {size} \nFilter: {filter} \nEpochs: {epochs} \nTest run: {small_dataset}")
        string_input = "pattern_spectra"
        if telescope_mode == "stereo_sum_cta":
            string_input_short = "_ps_float_alpha"
        elif telescope_mode == "mono":
            string_input_short = "_ps_float_mono_alpha"
        elif telescope_mode == "stereo_sum_ps":
            string_input_short = "_ps_float_stereo_sum_alpha"
        string_ps_input = f"a_{attribute[0]}_{attribute[1]}__dl_{domain_lower[0]}_{domain_lower[1]}__dh_{domain_higher[0]}_{domain_higher[1]}__m_{mapper[0]}_{mapper[1]}__n_{size[0]}_{size[1]}__f_{filter}/"
        string_table_column = "pattern spectrum"

    return(string_name, string_input, string_ps_input, string_input_short, string_table_column)

def PlotExamplesSeparation(X, Y, string_input, string_ps_input, string_name, mode):
    """
    Plot a few examples of the input data
    Args:
        X: input data
        Y: particle type
        string_input: string to indicate the input data
        string_ps_input: string to indicate the input data for pattern spectra
        string_name: string to indicate the name of the experiment
        mode: string to indicate the mode
    """
    # # plot a few examples
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
    plt.savefig(f"cnn/{string_input}/{mode}/results/{string_ps_input}/{string_name[1:]}/input_examples" + string_name + ".pdf", dpi = 250)
    plt.close()

def PlotEnergyDistributionSeparation(table, string_input, string_ps_input, string_name, mode):
    """
    Plot the energy distribution for the separation mode.
    Args:
        table: input data
        string_input: string to indicate the input data
        string_ps_input: string to indicate the input data for pattern spectra
        string_name: string to indicate the name of the experiment
        mode: string to indicate the mode
    """
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
    plt.savefig(f"cnn/{string_input}/{mode}/results/" + string_ps_input + f"{string_name[1:]}/" + "total_energy_distribution" + string_name + ".pdf", dpi = 250)
    plt.close()