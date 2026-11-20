"""A reconstruction basis: what a transient from this survey ought to look like.

The criterion needs only ``fit`` then ``reconstruct``, so any basis that offers
the two drops in without the step changing. Asif et al. (2022) use an
auto-encoder trained on forward responses of 1D models; this is the linear
stand-in, a low-rank basis over the gate dimension fitted on the delivered
survey itself.

Fitting on the delivered survey is its strength and its risk. The vendor's own
processing lands in the basis rather than in the residual, so it is not mistaken
for an anomaly. But a basis fitted on data containing coupling absorbs the
coupling if there is enough of it to reach the leading components.
``n_components`` is the knob: few enough to capture only the smooth decay.

Missing gates are ``NaN`` in the transients handed over: a gate that is culled,
non-positive, or past a sounding's noise cutoff. The basis is fitted by
alternating least squares over the known gates only, so every sounding
contributes the gates it has and nothing is imputed, and a sounding is
reconstructed by least squares on its own available gates. A gate known in too
few of the fitted soundings to determine its mean and loadings is left out of
the basis and reconstructs as ``NaN``, so the criterion does not judge it.
"""

from typing import Protocol

import numpy as np


class Reconstructor(Protocol):
    """Anything the reconstruction criterion can use as a basis."""

    def fit(self, transients: np.ndarray) -> None:
        """Learn the basis from ``(soundings, gates)`` transients; ``NaN`` is missing."""

    def reconstruct(self, transients: np.ndarray) -> np.ndarray:
        """Each transient as the basis would have it; ``NaN`` where it cannot say."""


class PCAReconstructor:
    """A low-rank basis over the gate dimension, in log amplitude, fitted on this survey.

    Parameters
    ----------
    n_components : int
        How many components to keep. Too many and the basis learns the coupling
        too; the default keeps the smooth decay and little else.
    log_space : bool
        Work in ``log10`` of the amplitude, where a decay is close to linear in
        the components. Non-positive values are then treated as missing.
    tolerance : float
        Fitting stops when the root-mean-square misfit of the basis to the
        known gates improves by less than this fraction between rounds.
    max_iterations : int
        Rounds of alternating least squares allowed; whether the fit converged
        is recorded in ``converged_``.
    max_soundings : int
        The basis is fitted on at most this many soundings, evenly spaced
        through the survey; every sounding is still reconstructed. A basis of
        a few components is well determined long before this.

    Attributes
    ----------
    gates_ : numpy.ndarray
        ``(gates,)`` bool after :meth:`fit`: which gates the basis covers. A
        gate known in no more of the fitted soundings than there are
        components is left out and reconstructs as ``NaN``.
    """

    def __init__(self, n_components: int = 4, log_space: bool = True,
                 tolerance: float = 1e-5, max_iterations: int = 200,
                 max_soundings: int = 20000):
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1; got {n_components}")
        self.n_components = int(n_components)
        self.log_space = bool(log_space)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.max_soundings = int(max_soundings)
        self.mean_ = None
        self.components_ = None
        self.gates_ = None
        self.iterations_ = 0
        self.converged_ = None

    def _forward(self, transients):
        values = np.asarray(transients, dtype=float)
        if not self.log_space:
            return values.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(values), np.nan)

    def _inverse(self, values):
        return 10.0 ** values if self.log_space else values

    def fit(self, transients: np.ndarray) -> None:
        """Fit the basis.

        Parameters
        ----------
        transients : numpy.ndarray
            ``(soundings, gates)`` amplitudes in the data's own units; ``NaN``
            where a gate is unavailable.

        Raises
        ------
        ValueError
            If no more soundings than components have any usable gate, or no
            gate is available in more soundings than that.
        """
        values = self._forward(transients)
        values = values[np.isfinite(values).any(axis=1)]
        if len(values) > self.max_soundings:
            values = values[np.linspace(0, len(values) - 1, self.max_soundings).astype(int)]
        if len(values) <= self.n_components:
            raise ValueError(
                f"{len(values)} sounding(s) with a usable gate; a basis of "
                f"{self.n_components} component(s) needs more.")
        known = np.isfinite(values)
        # A gate's mean and loadings are one unknown more than the components,
        # so a gate known in no more soundings than that cannot be fitted: the
        # same rule as for the soundings above. It is left out of the basis
        # and reconstruct() returns NaN for it; on a large survey the fitted
        # sample can miss a gate that only a few soundings have.
        self.gates_ = known.sum(axis=0) > self.n_components
        if not self.gates_.any():
            raise ValueError(
                f"No gate is available in more than {self.n_components} sounding(s); "
                f"a basis of {self.n_components} component(s) needs more.")
        values, known = values[:, self.gates_], known[:, self.gates_]

        # Start from the leading directions of the mean-filled data, then
        # alternate: each sounding's coefficients given the basis, then each
        # gate's mean and loadings given the coefficients, on known gates only.
        k = min(self.n_components, values.shape[1])
        mean = np.nanmean(values, axis=0)
        _, _, vt = np.linalg.svd(np.where(known, values, mean) - mean, full_matrices=False)
        components = vt[:k].copy()
        self.iterations_, self.converged_, misfit = 0, False, np.inf
        while not self.converged_ and self.iterations_ < self.max_iterations:
            coefficients = _solve_rows(components, values - mean, known)
            for gate in range(values.shape[1]):
                rows = known[:, gate]
                design = np.column_stack([np.ones(rows.sum()), coefficients[rows]])
                solution, *_ = np.linalg.lstsq(design, values[rows, gate], rcond=None)
                mean[gate], components[:, gate] = solution[0], solution[1:]
            residual = mean + coefficients @ components - values
            previous, misfit = misfit, np.sqrt(np.mean(residual[known] ** 2))
            self.converged_ = bool(previous - misfit <= self.tolerance * max(misfit, 1e-12))
            self.iterations_ += 1
        # An orthonormal basis spanning the same space, for stable projections.
        self.mean_ = mean
        self.components_ = np.linalg.qr(components.T)[0].T

    def reconstruct(self, transients: np.ndarray) -> np.ndarray:
        """Project each sounding onto the basis using only its available gates.

        Parameters
        ----------
        transients : numpy.ndarray
            ``(soundings, gates)`` amplitudes, as given to :meth:`fit`.

        Returns
        -------
        numpy.ndarray
            The reconstruction, the same shape, in the data's own units. A
            sounding with fewer available gates than components is ``NaN``
            throughout: there is nothing to constrain its coefficients. A gate
            the basis left out (``gates_``) is ``NaN`` in every sounding.
        """
        if self.components_ is None:
            raise RuntimeError("reconstruct() before fit()")
        values = self._forward(transients)
        out = np.full(values.shape, np.nan)
        values = values[:, self.gates_]
        known = np.isfinite(values)
        enough = known.sum(axis=1) >= len(self.components_)
        coefficients = _solve_rows(self.components_, values[enough] - self.mean_, known[enough])
        out[np.ix_(enough, self.gates_)] = self.mean_ + coefficients @ self.components_
        return self._inverse(out)


def _solve_rows(components, values, known):
    """Each row's coefficients on ``components``, by least squares on its known entries.

    Rows sharing the same known entries are solved together, which is most of
    them: the unknowns are a noise-cutoff tail and the occasional culled gate.
    """
    coefficients = np.zeros((len(values), len(components)))
    if not len(values):
        return coefficients
    patterns, group = np.unique(known, axis=0, return_inverse=True)
    group = np.asarray(group).ravel()
    for index, pattern in enumerate(patterns):
        if not pattern.any():
            continue
        rows = np.flatnonzero(group == index)
        solution, *_ = np.linalg.lstsq(components[:, pattern].T,
                                       values[np.ix_(rows, pattern)].T, rcond=None)
        coefficients[rows] = solution.T
    return coefficients
