import numpy as np
import emcee

def lnpdf(x, mu, icov):
	diff = x-mu
	return -np.dot(diff, np.dot(icov, diff))/2

ndim = 50

# mu
mu = np.random.rand(ndim)

# cov
cov = np.random.rand(ndim**2).reshape((ndim, ndim))-0.5
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov, cov)

# icov
icov = np.linalg.inv(cov)


nwalkers = 250
nsteps = 1000
p0 = np.random.rand(nwalkers*ndim).reshape((nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpdf, args=(mu, icov))
sampler.run_mcmc(p0, nsteps)

import matplotlib.pyplot as plt
plt.figure(facecolor='w', edgecolor='k')
for i in range(ndim):
	plt.subplot(ndim//5, 5, i+1)
	plt.xticks([])
	plt.yticks([])
	plt.hist(sampler.flatchain[:, i], 100, histtype='step')
plt.show()
