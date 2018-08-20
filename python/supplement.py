


    import pylab
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    prob = stats.probplot(np.cbrt(X_[:,1]), dist=stats.norm, plot=pylab)

    ax2 = fig.add_subplot(212)
    xt, _ = stats.boxcox(X_[:,1].ravel())
    prob = stats.probplot(xt, dist=stats.norm, plot=pylab)
