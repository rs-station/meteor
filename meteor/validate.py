"""map quality assessment: negentropy and negentropy maximization"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import differential_entropy

from .metadata import EvaluatedPoint
from .rsmap import Map
from .settings import MAP_SAMPLING


def negentropy(samples: np.ndarray, *, tolerance: float = 0.1) -> float:
    """
    Computes the negentropy of a given sample array.

    Negentropy is defined as the difference between the entropy of a given
    distribution and the entropy of a Gaussian distribution with the same variance.
    It is a measure of non-Gaussianity, with higher values indicating greater deviation
    from Gaussianity.

    Parameters
    ----------
    samples: np.ndarray
        A numpy array of sample data for which to calculate the negentropy.

    tolerance: float
        Tolerance level determining if the negentropy is suspiciously negative. Defaults to 0.1.

    Returns
    -------
    negentropy: float
        The negentropy of the distribution of `samples`.

    Raises
    ------
    ValueError: If the computed negentropy is less than the negative tolerance,
                indicating potential issues with the computation.

    References
    ----------
    http://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/

    Example:
    -------
    >>> samples = np.random.normal(size=1000)
    >>> negentropy(samples)
    0.0012  # Example output, varies with input samples.
    """
    std = np.std(samples.flatten())
    if std <= 0.0:
        return -np.inf

    neg_e = 0.5 * np.log(2.0 * np.pi * std**2) + 0.5 - differential_entropy(samples.flatten())
    if not neg_e >= -tolerance:
        msg = (
            f"negentropy is a large negative number {neg_e}, exceeds the tolerance {tolerance}"
            " -- something may have gone wrong"
        )
        raise ValueError(msg)

    return float(neg_e)


def map_negentropy(map_to_assess: Map, *, tolerance: float = 0.1) -> float:
    """
    Computes the negentropy of a crystallographic map.

    Negentropy is defined as the difference between the entropy of a given
    distribution and the entropy of a Gaussian distribution with the same variance.
    It is a measure of non-Gaussianity, with higher values indicating greater deviation
    from Gaussianity.

    Parameters
    ----------
    map_to_assess: Map
        The map (coefficients) we will compute the negentropy for.

    tolerance: float
        Tolerance level determining if the negentropy is suspiciously negative. Defaults to 0.1.

    Returns
    -------
    negentropy: float
        The negentropy of the distribution of the (implicitly computed) map voxels.

    Raises
    ------
    ValueError: If the computed negentropy is less than the negative tolerance,
                indicating potential issues with the computation.
    """
    realspace_map_array = map_to_assess.to_3d_numpy_map(map_sampling=MAP_SAMPLING)
    return negentropy(realspace_map_array, tolerance=tolerance)


class ScalarMaximizer:
    """
    Maximize a function using one of two strategies, or a combination of both:

      1. simply loop over an explicit list of argument values and pick the max
      2. use the Golden ratio method to try and be a bit faster

    The objective function to maximize must be a scalar mapping, taking a single float as an
    argument and returning a single float. In the context of METEOR, this is useful to e.g.
    maximize over simple parameter choices, such as TV regularization strength or k-weight values.

    Attributes
    ----------
    objective : Callable[[float], float]
        The objective function to maximize.
    argument_optimum : float
        The argument (input) that gives the highest objective value found so far.
    objective_maximum : float
        The maximum value of the objective function found so far.
    values_evaluated : set[float]
        A set of argument values that have been evaluated during optimization.
    """

    def __init__(self, *, objective: Callable[[float], float]) -> None:
        """
        Initializes the Maximizer with the given objective function.

        Parameters
        ----------
        objective : Callable[[float], float]
            The objective function to be maximized. It should map a single float to a float.
        """
        self.objective = objective
        self.argument_optimum: float = np.nan
        self.objective_maximum: float = -np.inf
        self.values_evaluated: list[float] = []
        self.objective_at_values: list[float] = []

    def _update_optima(self, argument_test_value: float) -> float:
        objective_value = float(self.objective(argument_test_value))
        if objective_value > self.objective_maximum:
            self.argument_optimum = argument_test_value
            self.objective_maximum = objective_value
        return float(objective_value)

    @property
    def parameter_scan_results(self) -> list[EvaluatedPoint]:
        return [
            EvaluatedPoint(parameter_value=pv, objective_value=ov)
            for pv, ov in zip(self.values_evaluated, self.objective_at_values, strict=False)
        ]

    def optimize_over_explicit_values(
        self, *, arguments_to_scan: Sequence[float] | np.ndarray
    ) -> None:
        """
        Scans through a list or array of argument values to find the optimum.

        Parameters
        ----------
        arguments_to_scan : list[float] | np.ndarray
            A list or array of argument values to evaluate.
        """
        for argument_test_value in arguments_to_scan:
            objective_value = self._update_optima(argument_test_value)
            self.values_evaluated.append(float(argument_test_value))
            self.objective_at_values.append(float(objective_value))

    def optimize_with_golden_algorithm(
        self,
        *,
        bracket: tuple[float, float],
        tolerance: float = 0.001,
    ) -> None:
        """
        Uses the golden-section search algorithm to maximize the objective function within a given
        bracket.

        Parameters
        ----------
        bracket : tuple[float, float]
            A tuple containing the lower and upper bounds of the bracket within which to search
            for the optimum.

        Raises
        ------
        RuntimeError
            If the optimization using the golden algorithm fails.
        """

        def _objective_with_value_tracking(argument_test_value: float) -> float:
            """Adds the evaluated value to self.values_evaluated"""
            objective_value = self.objective(argument_test_value)
            self.values_evaluated.append(argument_test_value)
            self.objective_at_values.append(objective_value)
            return -objective_value  # negative: we want max

        optimizer_result = minimize_scalar(
            _objective_with_value_tracking,
            bracket=bracket,
            method="golden",
            tol=tolerance,
        )
        if not optimizer_result.success:
            msg = "Golden minimization failed"
            raise RuntimeError(msg)

        self._update_optima(optimizer_result.x)
