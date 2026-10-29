"""A reconstruction basis: what a transient from this survey ought to look like.

The criterion needs only ``fit`` then ``reconstruct``, so any basis that offers
the two drops in without the step changing. Asif et al. (2022) use an
auto-encoder trained on forward responses of 1D models; this is the linear
stand-in, an SVD over the gate dimension fitted on the delivered survey itself.

Fitting on the delivered survey is its strength and its risk. The vendor's own
processing lands in the basis rather than in the residual, so it is not mistaken
for an anomaly. But a basis fitted on data containing coupling absorbs the
coupling if there is enough of it to reach the leading components.
``n_components`` is the knob: few enough to capture only the smooth decay.

Missing gates are ``NaN`` in the transients handed over: a gate that is culled,
non-positive, or past a sounding's noise cutoff. The basis is fitted by
iterative SVD imputation, so every sounding contributes the gates it has, and a
sounding is reconstructed by least squares on its own available gates.
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
    """SVD over the gate dimension, in log amplitude, fitted on this survey.

    Parameters
    ----------
    n_components : int
        How many components to keep. Too many and the basis learns the coupling
        too; the default keeps the smooth decay and little else.
    log_space : bool
        Work in ``log10`` of the amplitude, where a decay is close to linear in
        the components. Non-positive values are then treated as missing.
    tolerance : float
        Imputation stops when no missing value moves by more than this between
        rounds, in the units the basis is fitted in (``log10`` amplitude by
        default, so ``1e-7`` is a relative change of about 0.2 ppm).
    max_iterations : int
        Rounds of imputation allowed. Convergence is linear, so a survey with
        many missing gates can need hundreds; whether it converged is recorded
        in ``converged_``.
    """

    def __init__(self, n_components: int = 4, log_space: bool = True,
                 tolerance: float = 1e-7, max_iterations: int = 500):
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1; got {n_components}")
        self.n_components = int(n_components)
        self.log_space = bool(log_space)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.mean_ = None
        self.components_ = None
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
            If fewer soundings than components have any usable gate, or a gate
            is unavailable in every sounding.
        """
        values = self._forward(transients)
        rows = np.isfinite(values).any(axis=1)
        values = values[rows]
        if len(values) <= self.n_components:
            raise ValueError(
                f"{len(values)} sounding(s) with a usable gate; a basis of "
                f"{self.n_components} component(s) needs more.")
        known = np.isfinite(values)
        if not known.any(axis=0).all():
            raise ValueError(
                "At least one gate is unavailable in every sounding; leave it "
                "out of the transients rather than fit a basis over it.")

        filled = np.where(known, values, np.nanmean(values, axis=0))
        k = min(self.n_components, values.shape[1])
        self.iterations_, self.converged_ = 0, known.all()
        while not self.converged_ and self.iterations_ < self.max_iterations:
            mean = filled.mean(axis=0)
            _, _, vt = np.linalg.svd(filled - mean, full_matrices=False)
            components = vt[:k]
            approximation = mean + (filled - mean) @ components.T @ components
            update = np.where(known, values, approximation)
            self.converged_ = bool(np.abs(update - filled).max() <= self.tolerance)
            filled = update
            self.iterations_ += 1
        self.mean_ = filled.mean(axis=0)
        _, _, vt = np.linalg.svd(filled - self.mean_, full_matrices=False)
        self.components_ = vt[:k]

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
            throughout: there is nothing to constrain its coefficients.
        """
        if self.components_ is None:
            raise RuntimeError("reconstruct() before fit()")
        values = self._forward(transients)
        out = np.full(values.shape, np.nan)
        k = len(self.components_)
        for row in range(len(values)):
            known = np.isfinite(values[row])
            if known.sum() < k:
                continue
            basis = self.components_[:, known].T
            coefficients, *_ = np.linalg.lstsq(
                basis, values[row, known] - self.mean_[known], rcond=None)
            out[row] = self.mean_ + coefficients @ self.components_
        return self._inverse(out)
