
"""This module defines multiple implementations of the abstract class
:py:class:`StochasticModel <dowhy.scm.graph.StochasticModel>`
"""

import warnings
from typing import Union, Tuple, Dict, Optional

import numpy as np
import scipy
from scipy.stats import rv_continuous, rv_discrete, norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import BayesianGaussianMixture

from dowhy.gcm.divergence import estimate_kl_divergence_continuous
from dowhy.gcm.graph import StochasticModel
from dowhy.gcm.util.general import shape_into_2d

_CONTINUOUS_DISTRIBUTIONS = [
    scipy.stats.norm,
    scipy.stats.laplace,
    scipy.stats.t,
    scipy.stats.uniform,
    scipy.stats.rayleigh
]
_CONTINUOUS_DISTRIBUTIONS.extend(
    [getattr(scipy.stats, d) for d in dir(scipy.stats) if isinstance(getattr(scipy.stats, d), scipy.stats.rv_continuous)
     and d not in _CONTINUOUS_DISTRIBUTIONS])

_DISCRETE_DISTRIBUTIONS = [getattr(scipy.stats, d) for d in dir(scipy.stats) if isinstance(getattr(scipy.stats, d),
                                                                                           scipy.stats.rv_discrete)]

_CONTINUOUS_DISTRIBUTIONS = {x.name: x for x in _CONTINUOUS_DISTRIBUTIONS}
_DISCRETE_DISTRIBUTIONS = {x.name: x for x in _DISCRETE_DISTRIBUTIONS}


class ScipyDistribution(StochasticModel):

    def __init__(self,
                 scipy_distribution: Optional[Union[rv_continuous, rv_discrete]] = None,
                 **parameters) -> None:
        self.__distribution = scipy_distribution
        self.__parameters = parameters
        self.__fixed_parameters = len(parameters) > 0

    def draw_samples(self, num_samples: int) -> np.ndarray:
        if len(self.__parameters) == 0 or self.__distribution is None:
            raise ValueError('Cannot draw samples. Model has not been fit!')

        return shape_into_2d(self.__distribution.rvs(size=num_samples,
                                                     **self.parameters))

    def fit(self, X: np.ndarray) -> None:
        if self.__distribution is None:
            # Currently only support continuous distributions for auto selection.
            best_model, best_parameters = self.find_suitable_continuous_distribution(X)
            self.__distribution = best_model
            self.__parameters = best_parameters
        elif not self.__fixed_parameters:
            self.__parameters \
                = self.map_scipy_distribution_parameters_to_names(self.__distribution,
                                                                  self.__distribution.fit(shape_into_2d(X)))

    @property
    def parameters(self) -> Dict[str, float]:
        return self.__parameters

    @property
    def scipy_distribution(self) -> Optional[Union[rv_continuous, rv_discrete]]:
        return self.__distribution

    def clone(self):
        if self.__fixed_parameters:
            return ScipyDistribution(scipy_distribution=self.__distribution, **self.__parameters)
        else:
            return ScipyDistribution(scipy_distribution=self.__distribution)

    @staticmethod
    def find_suitable_continuous_distribution(distribution_samples: np.ndarray,
                                              divergence_threshold: float = 10 ** -2) \
            -> Tuple[rv_continuous, Dict[str, float]]:
        """ Tries to find the best fitting continuous parametric distribution of given samples. """
        distribution_samples = shape_into_2d(distribution_samples)

        currently_best_distribution = norm
        currently_best_parameters = (0.0, 1.0)
        currently_smallest_divergence = np.inf

        # Estimate distribution parameters from data.
        for distribution in _CONTINUOUS_DISTRIBUTIONS.values():
            # Ignore warnings from fitting process.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                try:
                    # Fit distribution to data.
                    params = distribution.fit(distribution_samples)
                except ValueError:
                    # Some distributions might not be compatible with the data.
                    continue

                # Separate parts of parameters.
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                generated_samples = distribution.rvs(size=distribution_samples.shape[0],
                                                     loc=loc,
                                                     scale=scale,
                                                     *arg)

                # Check the KL divergence between the distribution of the given and fitted distribution.
                divergence \
                    = estimate_kl_divergence_continuous(distribution_samples, generated_samples)
                if divergence < divergence_threshold:
                    currently_best_distribution = distribution
                    currently_best_parameters = params
                    break

                # Identify if this distribution is better.
                if currently_smallest_divergence > divergence:
                    currently_best_distribution = distribution
                    currently_best_parameters = params
                    currently_smallest_divergence = divergence

        return currently_best_distribution, \
               ScipyDistribution.map_scipy_distribution_parameters_to_names(currently_best_distribution,
                                                                            currently_best_parameters)

    @staticmethod
    def map_scipy_distribution_parameters_to_names(scipy_distribution: Union[rv_continuous, rv_discrete],
                                                   parameters: Tuple[float]) -> Dict[str, float]:
        if scipy_distribution.shapes:
            parameter_list = [name.strip() for name in scipy_distribution.shapes.split(',')]
        else:
            parameter_list = []
        if scipy_distribution.name in _DISCRETE_DISTRIBUTIONS:
            parameter_list += ['loc']
        elif scipy_distribution.name in _CONTINUOUS_DISTRIBUTIONS:
            parameter_list += ['loc', 'scale']
        else:
            raise ValueError("Distribution %s not found in the list of continuous and discrete distributions!"
                             % scipy_distribution.name)

        parameters_dictionary = {}
        for i, parameter_name in enumerate(parameter_list):
            parameters_dictionary[parameter_name] = parameters[i]

        return parameters_dictionary


class EmpiricalDistribution(StochasticModel):
    """ A distribution model for uniformly sampling from data samples. """

    def __init__(self) -> None:
        self.__data = None

    @property
    def data(self) -> np.ndarray:
        return self.__data

    def fit(self, X: np.ndarray) -> None:
        self.__data = shape_into_2d(X)

    def draw_samples(self, num_samples: int) -> np.ndarray:
        if self.data is None:
            raise RuntimeError('%s has not been fitted!' % self.__class__.__name__)

        return self.data[np.random.choice(self.data.shape[0], size=num_samples, replace=True), :]

    def clone(self):
        return EmpiricalDistribution()


class BayesianGaussianMixtureDistribution(StochasticModel):
    def __init__(self) -> None:
        self.__gmm_model = None

    def fit(self,
            X: np.ndarray) -> None:
        X = shape_into_2d(X)
        self.__gmm_model = BayesianGaussianMixture(
            n_components=BayesianGaussianMixtureDistribution.__get_optimal_number_of_components(X),
            max_iter=1000).fit(X)

    @staticmethod
    def __get_optimal_number_of_components(X: np.ndarray) -> int:
        current_best = 0
        current_best_num_components = 1
        num_best_in_succession = 0
        try:
            for i in range(2, int(np.sqrt(X.shape[0] / 2))):
                kmeans = KMeans(n_clusters=i).fit(X)
                coefficient = silhouette_score(X, kmeans.labels_, sample_size=5000)

                if coefficient > current_best:
                    current_best = coefficient
                    current_best_num_components = i
                    num_best_in_succession = 0
                else:
                    num_best_in_succession += 1

                if num_best_in_succession >= 3:
                    break
        except ValueError:
            # This error is typically raised when the data is discrete and all points are assigned to less cluster than
            # specified. It can also happen due to duplicated points. In these cases, the current best solution should
            # be sufficient.
            return current_best_num_components

        return current_best_num_components

    def draw_samples(self, num_samples: int) -> np.ndarray:
        if self.__gmm_model is None:
            raise RuntimeError('%s has not been fitted!' % self.__class__.__name__)

        return shape_into_2d(self.__gmm_model.sample(num_samples)[0])

    def __str__(self) -> str:
        return 'Approximated data distribution'

    def clone(self):
        return BayesianGaussianMixtureDistribution()
