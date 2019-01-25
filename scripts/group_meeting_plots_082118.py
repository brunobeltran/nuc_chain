#plotting parameters
import matplotlib.cm as cm
import scipy
from scipy.optimize import curve_fit

params = {'axes.edgecolor': 'black', 'axes.facecolor': 'white', 'axes.grid': False,
'axes.linewidth': 0.75, 'backend': 'pdf','axes.labelsize': 12,'legend.fontsize': 10,
'xtick.labelsize': 10,'ytick.labelsize': 10,'text.usetex': False,'figure.figsize': [7, 5],
'mathtext.fontset': 'stixsans', 'savefig.format': 'pdf', 'xtick.major.pad': 4, 'xtick.major.size': 5, 'xtick.major.width': 0.5,
'ytick.major.pad': 4, 'ytick.major.size': 5, 'ytick.major.width': 0.5,}

plt.rcParams.update(params)

def plot_kuhn_length_colored_by_period():
    fig, ax = plt.subplots()
    plt.plot(np.arange(1, 251), kuhns1to250[:,0], linewidth=0.5)
    plt.scatter(np.arange(1, 251), kuhns1to250[:,0], c=np.arange(1.5,251.5)%10.5, cmap='Spectral')
    plt.xlabel('Linker length (bp)')
    plt.ylabel('Kuhn Length (nm)')
    plt.title(r'$b=\langle{R^2}\rangle/R_{max}$, colored by (linker length % 10.5bp)')
    plt.colorbar()
    plt.subplots_adjust(bottom=0.12, right=1.0, top=0.9)

def plot_kuhn_length_in_bp():
    kuhns_in_bp = wlc.tabulate_kuhn_lengths_in_bp()
    plt.rcParams.update({'figure.figsize':[6, 5]})
    kuhns_bp_period_no_unwraps = kuhns_in_bp[35:47, 0]
    kuhns_bp_period_20_unwraps = kuhns_in_bp[35:47, 20]
    fig, ax = plt.subplots()
    plt.xlabel('Linker length (bp)')
    plt.ylabel('Kuhn length (bp)')
    ax.plot(np.arange(36, 48), kuhns_bp_period_no_unwraps, 'g-o', label='no unwraps')
    ax.plot(np.arange(36, 48), kuhns_bp_period_20_unwraps, 'r-o', label='20bp unwraps', markersize=4, linewidth=1)
    plt.legend()
    plt.savefig('plots/kuhn/kuhn_length_in_bp_36to47links_0and20unwraps.png')

def plot_angle_screw_axis_vs_linkers():
    plt.rcParams.update({'figure.figsize':[5.25, 3.75]})
    link_ix, unwrap_ix, rise, angle, radius = ncg.tabulate_rise()
    fig, ax = plt.subplots()
    plt.plot(np.arange(10, 200), angle[:,0], linewidth=0.5)
    plt.scatter(np.arange(10, 200), angle[:,0], c=np.arange(10,200)%10.5, cmap='Spectral');
    plt.xlabel('Linker length (bp)')
    plt.ylabel(r'$\theta$')
    plt.title('Angle of rotation about screw-axis')
    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.92, right=1.0)
    plt.colorbar()
    plt.savefig('plots/kuhn/angle_screw_axis_vs_linkers10to200_colored_by_period.png')

def scatter_kuhn_length_vs_angle():
    plt.rcParams.update({'figure.figsize':[5.25, 3.75]})
    fig, ax = plt.subplots()
    plt.scatter(angle[:,0], kuhns1to250[9:199,0], c=np.arange(0, 190)%10.5, cmap ='Spectral')
    plt.ylabel('Kuhn length (nm)')
    plt.xlabel(r'$\theta$')
    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.92, right=1.0)
    plt.colorbar()
    plt.savefig('plots/kuhn/scatter_kuhn_vs_angle_colored_by_period.png')


def f(x, a, b, k):
    return a*np.exp(-k*x) + b

def kuhn_length_vs_angle_exponential_fit():
    #There are 9 periods represented above (ignore last point -- start of 10th period)
    #9 dots per vertical line, 21 vertical lines for 21 different helices)
    #Programmatically extract points on each of these curves

    peak_start = 10
    periods = np.arange(0, 9)
    #store parameters extracted by curvefit for each of the curves
    exp_params = np.zeros((len(periods), 3))
    residuals = np.zeros((len(periods), 21))

    plt.rcParams.update({'figure.figsize':[5.25, 3.75]})
    fig, ax = plt.subplots()
    plt.xlabel('Angle of rotation about screw-axis (rad)')
    plt.ylabel('Kuhn length (nm)')
    angdata = np.linspace(min(angle[:,0]), max(angle[:,0]), 100)

    for i in periods:
        links_peak_to_peak = np.arange(peak_start, peak_start+21)
        kuhns = kuhns1to250[links_peak_to_peak-1, 0]
        angles = angle[links_peak_to_peak-10, 0]
        popt, pcov = curve_fit(f, angles, kuhns, bounds=[[-np.inf, 70, 0.1], [0, 100, 5]])
        exp_params[i, :] = popt
        plt.scatter(angle[:,0], kuhns1to250[9:199,0], c=np.arange(10, 200), cmap ='viridis')
        plt.plot(angdata, f(angdata, *popt), 'k', linewidth=0.5)
        residuals[i] = f(angles, *popt) - kuhns
        peak_start += 21

    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.93, right=0.98)
    plt.title('Exponential fit: $a*e^{-kx}+b$')
    plt.colorbar(label="Linker length")
    plt.savefig('plots/kuhn/exponential_fit_kuhn_vs_angle_colored_by_linker.png')
    return exp_params, residuals


def plot_exponential_parameters(exp_params):
    plt.rcParams.update({'figure.figsize':[5.5, 6]})
    peak_start = 10
    periods = np.arange(0, 9)
    linkers_in_periods = np.zeros_like(periods)
    for i in periods:
        linkers_in_periods[i] = 21*(i+1) + peak_start

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
    ax1.plot(linkers_in_periods, exp_params[:, 2], '-o')
    plt.xlabel('Linker length (bp)')
    ax1.set_ylabel(r'Decay constant $k$')
    ax2.plot(linkers_in_periods, exp_params[:, 0], '-o')
    ax2.set_ylabel(r'Scaling factor $a$')
    ax2.tick_params(axis='y', which='major', pad=2)
    plt.subplots_adjust(left=0.15, bottom=0.15, top=0.93, right=0.95)
    plt.savefig('plots/kuhn/exponential_params_kuhn_vs_angle.png')



