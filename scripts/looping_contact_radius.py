import numpy as np
from MultiPoint import propagator
from nuc_chain import fluctuations as wlc

Klin = np.linspace(0, 10**5, 20000)
Klog = np.logspace(-3, 5, 10000)
Kvals = np.unique(np.concatenate((Klin, Klog)))
#convert to little k -- units of inverse bp (this results in kmax = 332)
kvals = Kvals / (2*wlc.default_lp)
links35 = np.tile(35, 44)
Rlinks = np.array([47, 21, 18, 15, 20, 17])
Llinks = np.array([245, 30, 26, 15, 23, 35])

#links to right of methylation site
Rlinks = np.concatenate((Rlinks, links35))
#links to left of methylation site
Llinks = np.concatenate((Llinks, links35))
unwrap = 0
Kprops_bareWLC = wlc.tabulate_bareWLC_propagators(Kvals)
qprop_R = wlc.bareWLC_gprop(kvals, Rlinks, unwrap, props=Kprops_bareWLC)
qintegral_R = wlc.BRN_fourier_integrand_splines(kvals, Rlinks, unwrap, Bprop=qprop_R, rvals=rvals) #default: 1000 rvals
#BARE WLC LOOPING PROBABILITY CALCULATIONS
qprop_L = wlc.bareWLC_gprop(kvals, Llinks, unwrap, props=Kprops_bareWLC)
qintegral_L = wlc.BRN_fourier_integrand_splines(kvals, Llinks, unwrap, Bprop=qprop_L, rvals=rvals) #default: 1000 rvals

