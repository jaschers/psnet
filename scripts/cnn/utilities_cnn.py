from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, MaxPool2D, ReLU, Dropout, LeakyReLU

def ResBlock(z, kernelsizes, filters, increase_dim = False):
    # https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
    # https://stackoverflow.com/questions/64792460/how-to-code-a-residual-block-using-two-layers-of-a-basic-cnn-algorithm-built-wit
    # https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33

    z_shortcut = z
    kernelsize_1, kernelsize_2 = kernelsizes
    filters_1, filters_2 = filters

    fz = Conv2D(filters_1, kernelsize_1)(z)
    # fz = BatchNormalization()(fz)
    fz = ReLU()(fz)

    fz = Conv2D(filters_1, kernelsize_2, padding = "same")(fz)
    # fz = BatchNormalization()(fz)
    fz = ReLU()(fz)
    
    fz = Conv2D(filters_2, kernelsize_1)(fz)
    # fz = BatchNormalization()(fz)

    if increase_dim == True:
        z_shortcut = Conv2D(filters_2, (1, 1))(z_shortcut)
        # z_shortcut = BatchNormalization()(z_shortcut)

    out = Add()([fz, z_shortcut])
    out = ReLU()(out)
    # out = MaxPooling2D(pool_size=(3, 3), strides = 1)(out)
    
    return out

def PlotExamplesEnergy(X, path):
    # plot a few examples
    fig, ax = plt.subplots(3, 3)
    ax = ax.ravel()
    for i in range(9):
        ax[i].imshow(X[i], cmap = "Greys_r")
        ax[i].title.set_text(f"{int(np.round(10**Y[i]))} GeV")
        ax[i].axis("off")
    plt.savefig(path, dpi = 250)
    plt.close()

def EnergyDistributionEnergy(Y, path):
    plt.figure()
    plt.hist(10**Y, bins=np.logspace(np.log10(np.min(10**Y)),np.log10(np.max(10**Y)), 50))
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Number of events")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(path, dpi = 250)
    plt.close()