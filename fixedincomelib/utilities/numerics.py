import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6

    Convention
    ----------
    Node i "owns" the half-open bucket (x_{i-1}, x_i], so that f is left
    continuous: f(x_i) = y_i and f(x_i + eps) = y_{i+1}.  Flat extrapolation
    extends the first bucket to -inf and the last bucket to +inf:

        bucket_0     = (-inf, x_0]
        bucket_i     = (x_{i-1}, x_i]          for 0 < i < N-1
        bucket_{N-1} = (x_{N-2}, +inf)

    (for N == 1 the single bucket is the whole real line).

    Because f is linear in the ordinates y, the interpolated value and its
    integral are both of the form  sum_i w_i(x) * y_i , so the gradient with
    respect to y is just the weight vector w(x), which never depends on y.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT
        assert self.length_ >= 1, "need at least one knot"
        # lower / upper edge of every bucket (see class docstring)
        self.bucket_lo_ = np.concatenate(([-np.inf], self.axis1_[:-1]))
        self.bucket_hi_ = np.concatenate((self.axis1_[:-1], [np.inf]))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _bucket_index(self, x: float) -> int:
        """Index i of the bucket (x_{i-1}, x_i] containing x.

        np.searchsorted(..., side='left') returns the number of knots strictly
        smaller than x, which is exactly the left-continuous bucket index; the
        right wing (x > x_{N-1}) is clipped onto the last node.
        """
        idx = int(np.searchsorted(self.axis1_, x, side='left'))
        return min(idx, self.length_ - 1)

    def _bucket_overlaps(self, start_x: float, end_x: float) -> np.ndarray:
        """Length of  [start_x, end_x] intersect bucket_i  for every i (>= 0)."""
        lo = np.maximum(self.bucket_lo_, start_x)
        hi = np.minimum(self.bucket_hi_, end_x)
        return np.maximum(hi - lo, 0.0)

    # ------------------------------------------------------------------ #
    # interface
    # ------------------------------------------------------------------ #
    def interpolate(self, x: float) -> float:
        return float(self.values_[self._bucket_index(x)])

    def integrate(self, start_x: float, end_x: float) -> float:
        # int_l^u f = sum_i y_i * |[l,u] cap bucket_i| ; orientation-aware
        sign = 1.0
        if end_x < start_x:
            start_x, end_x = end_x, start_x
            sign = -1.0
        w = self._bucket_overlaps(start_x, end_x)
        return sign * float(np.dot(w, self.values_))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        # f(x) = y_i for the owning bucket  =>  df/dy = e_i
        grad = np.zeros(self.length_)
        grad[self._bucket_index(x)] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        # I = sum_i y_i * w_i  =>  dI/dy_i = w_i  (the overlap lengths)
        sign = 1.0
        if end_x < start_x:
            start_x, end_x = end_x, start_x
            sign = -1.0
        return sign * self._bucket_overlaps(start_x, end_x)

class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')