# pylint: disable=unused-argument
import numpy as np
import xarray as xr
from scipy import stats

from .family import Family


class UnivariateFamily(Family):
    def predict(self, model, posterior, linear_predictor):
        """Predict mean response"""
        response_var = model.response.name + "_mean"
        response_dim = model.response.name + "_obs"

        # Drop var/dim if already present
        if response_var in posterior.data_vars:
            posterior = posterior.drop_vars(response_var)

        if response_dim in posterior.dims:
            posterior = posterior.drop_dims(response_dim)

        posterior[response_var] = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        return posterior


class Bernoulli(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "logit", "probit", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        "Sample from posterior predictive distribution"
        mean = mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        return xr.apply_ufunc(np.random.binomial, 1, mean)


class Beta(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "logit", "probit", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        kappa = posterior[model.response.name + "_kappa"]
        alpha = mean * kappa
        beta = (1 - mean) * kappa
        return xr.apply_ufunc(np.random.beta, alpha, beta)


class Binomial(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "logit", "probit", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor, trials=None):
        if trials is None:
            trials = model.response.data[:, 1]
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        return xr.apply_ufunc(np.random.binomial, trials.squeeze(), mean)


class Gamma(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        alpha = posterior[model.response.name + "_alpha"]
        beta = alpha / mean
        return xr.apply_ufunc(np.random.gamma, alpha, 1 / beta)


class Gaussian(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        "Sample from posterior predictive distribution"
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        sigma = posterior[model.response.name + "_sigma"]
        return xr.apply_ufunc(np.random.normal, mean, sigma)


class NegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        n = posterior[model.response.name + "_alpha"]
        p = n / (mean + n)
        return xr.apply_ufunc(np.random.negative_binomial, n, p)


class Laplace(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        "Sample from posterior predictive distribution"
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        b = posterior[model.response.name + "_b"]
        return xr.apply_ufunc(np.random.laplace, mean, b)


class Poisson(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        return xr.apply_ufunc(np.random.poisson, mean)


class StudentT(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        sigma = posterior[model.response.name + "_sigma"]

        if isinstance(self.likelihood.priors["nu"], (int, float)):
            nu = self.likelihood.priors["nu"]
        else:
            nu = posterior[model.response.name + "_nu"]

        return xr.apply_ufunc(stats.t.rvs, nu, mean, sigma)


class VonMises(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "tan_2"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        kappa = posterior[model.response.name + "_kappa"]
        return xr.apply_ufunc(np.random.vonmises, mean, kappa)


class Wald(UnivariateFamily):
    SUPPORTED_LINKS = ["inverse", "inverse_squared", "identity", "log"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        lam = posterior[model.response.name + "_lam"]
        return xr.apply_ufunc(np.random.wald, mean, lam)
