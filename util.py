import jax
import jax.numpy.fft
import jax.numpy as jnp
from jax import lax
from jax._src import core
from jax._src import dtypes
from jax._src.api_util import _ensure_index_tuple
from jax._src.lax.lax import PrecisionLike
from jax._src.numpy import linalg
from jax._src.numpy.util import (
    check_arraylike, promote_dtypes_inexact, promote_dtypes_complex)
from jax._src.third_party.scipy import signal_helper
from jax._src.typing import Array, ArrayLike
from jax._src.util import canonicalize_axis, tuple_delete, tuple_insert
import numpy as np
import math
def _overlap_and_add(x: Array, step_size: int) -> Array:
  """Utility function compatible with tf.signal.overlap_and_add.

  Args:
    x: An array with `(..., frames, frame_length)`-shape.
    step_size: An integer denoting overlap offsets. Must be less than
      `frame_length`.

  Returns:
    An array with `(..., output_size)`-shape containing overlapped signal.
  """
  check_arraylike("_overlap_and_add", x)
  step_size = core.concrete_or_error(
      int, step_size, "step_size for overlap_and_add")
  if x.ndim < 2:
    raise ValueError('Input must have (..., frames, frame_length) shape.')

  *batch_shape, nframes, segment_len = x.shape
  flat_batchsize = math.prod(batch_shape)
  x = x.reshape((flat_batchsize, nframes, segment_len))
  output_size = step_size * (nframes - 1) + segment_len
  nstep_per_segment = 1 + (segment_len - 1) // step_size

  # Here, we use shorter notation for axes.
  # B: batch_size, N: nframes, S: nstep_per_segment,
  # T: segment_len divided by S

  padded_segment_len = nstep_per_segment * step_size
  x = jnp.pad(x, ((0, 0), (0, 0), (0, padded_segment_len - segment_len)))
  x = x.reshape((flat_batchsize, nframes, nstep_per_segment, step_size))

  # For obtaining shifted signals, this routine reinterprets flattened array
  # with a shrinked axis.  With appropriate truncation/ padding, this operation
  # pushes the last padded elements of the previous row to the head of the
  # current row.
  # See implementation of `overlap_and_add` in Tensorflow for details.
  x = x.transpose((0, 2, 1, 3))  # x: (B, S, N, T)
  x = jnp.pad(x, ((0, 0), (0, 0), (0, nframes), (0, 0)))  # x: (B, S, N*2, T)
  shrinked = x.shape[2] - 1
  x = x.reshape((flat_batchsize, -1))
  x = x[:, :(nstep_per_segment * shrinked * step_size)]
  x = x.reshape((flat_batchsize, nstep_per_segment, shrinked * step_size))

  # Finally, sum shifted segments, and truncate results to the output_size.
  x = x.sum(axis=1)[:, :output_size]
  return x.reshape(tuple(batch_shape) + (-1,))

def istft(Zxx: Array, fs: ArrayLike = 1.0, window: str = 'hann',
          nperseg: int | None = None, noverlap: int | None = None,
          nfft: int | None = None, input_onesided: bool = True,
          boundary: bool = True, time_axis: int = -1,
          freq_axis: int = -2) -> tuple[Array, Array]:
  """
  Perform the inverse short-time Fourier transform (ISTFT).

  JAX implementation of :func:`scipy.signal.istft`; computes the inverse of
  :func:`jax.scipy.signal.stft`.

  Args:
    Zxx: STFT of the signal to be reconstructed.
    fs: Sampling frequency of the time series (default: 1.0)
    window: Data tapering window to apply to each segment. Can be a window function name,
      a tuple specifying a window length and function, or an array (default: ``'hann'``).
    nperseg: Number of data points per segment in the STFT. If ``None`` (default), the
      value is determined from the size of ``Zxx``.
    noverlap: Number of points to overlap between segments (default: ``nperseg // 2``).
    nfft: Number of FFT points used in the STFT. If ``None`` (default), the
      value is determined from the size of ``Zxx``.
    input_onesided: If Tru` (default), interpret the input as a one-sided STFT
      (positive frequencies only). If False, interpret the input as a two-sided STFT.
    boundary: If True (default), it is assumed that the input signal was extended at
      its boundaries by ``stft``. If `False`, the input signal is assumed to have been truncated at the boundaries by `stft`.
    time_axis: Axis in `Zxx` corresponding to time segments (default: -1).
    freq_axis: Axis in `Zxx` corresponding to frequency bins (default: -2).

  Returns:
    A length-2 tuple of arrays ``(t, x)``. ``t`` is the Array of signal times, and ``x``
    is the reconstructed time series.

  See Also:
    :func:`jax.scipy.signal.stft`: short-time Fourier transform.

  Examples:
    Demonstrate that this gives the inverse of :func:`~jax.scipy.signal.stft`:

    >>> x = jnp.array([1., 2., 3., 2., 1., 0., 1., 2.])
    >>> f, t, Zxx = jax.scipy.signal.stft(x, nperseg=4)
    >>> print(Zxx)  # doctest: +SKIP
    [[ 1. +0.j   2.5+0.j   1. +0.j   1. +0.j   0.5+0.j ]
     [-0.5+0.5j -1.5+0.j  -0.5-0.5j -0.5+0.5j  0. -0.5j]
     [ 0. +0.j   0.5+0.j   0. +0.j   0. +0.j  -0.5+0.j ]]
    >>> t, x_reconstructed = jax.scipy.signal.istft(Zxx)
    >>> print(x_reconstructed)
    [1. 2. 3. 2. 1. 0. 1. 2.]
  """
  # Input validation
  check_arraylike("istft", Zxx)
  if Zxx.ndim < 2:
    raise ValueError('Input stft must be at least 2d!')
  freq_axis = canonicalize_axis(freq_axis, Zxx.ndim)
  time_axis = canonicalize_axis(time_axis, Zxx.ndim)
  if freq_axis == time_axis:
    raise ValueError('Must specify differing time and frequency axes!')

  Zxx = jnp.asarray(Zxx, dtype=jax.dtypes.canonicalize_dtype(
      np.result_type(Zxx, np.complex64)))

  n_default = (2 * (Zxx.shape[freq_axis] - 1) if input_onesided
               else Zxx.shape[freq_axis])

  nperseg_int = core.concrete_or_error(int, nperseg or n_default,
                                           "nperseg: segment length of STFT")
  if nperseg_int < 1:
    raise ValueError('nperseg must be a positive integer')

  nfft_int: int = 0
  if nfft is None:
    nfft_int = n_default
    if input_onesided and nperseg_int == n_default + 1:
      nfft_int += 1  # Odd nperseg, no FFT padding
  else:
    nfft_int = core.concrete_or_error(int, nfft, "nfft of STFT")
  if nfft_int < nperseg_int:
    raise ValueError(
        f'FFT length ({nfft_int}) must be longer than nperseg ({nperseg_int}).')

  noverlap_int = core.concrete_or_error(
      int, noverlap or nperseg_int // 2, "noverlap of STFT")
  if noverlap_int >= nperseg_int:
    raise ValueError('noverlap must be less than nperseg.')
  nstep = nperseg_int - noverlap_int

  # Rearrange axes if necessary
  if time_axis != Zxx.ndim - 1 or freq_axis != Zxx.ndim - 2:
    outer_idxs = tuple(
        idx for idx in range(Zxx.ndim) if idx not in {time_axis, freq_axis})
    Zxx = jnp.transpose(Zxx, outer_idxs + (freq_axis, time_axis))

  # Perform IFFT
  ifunc = jax.numpy.fft.irfft if input_onesided else jax.numpy.fft.ifft
  # xsubs: [..., T, N], N is the number of frames, T is the frame length.
  xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg_int, :]

  # Get window as array
  if window == 'hann':
    # Implement the default case without scipy
    win = jnp.array([1.0]) if nperseg_int == 1 else jnp.sin(jnp.linspace(0, jnp.pi, nperseg_int, endpoint=False)) ** 2
    win = win.astype(xsubs.dtype)
  elif isinstance(window, (str, tuple)):
    # TODO(jakevdp): implement get_window() in JAX to remove optional scipy dependency
    try:
      from scipy.signal import get_window
    except ImportError as err:
      raise ImportError(f"scipy must be available to use {window=}") from err
    win = get_window(window, nperseg_int)
    win = jnp.array(win, dtype=xsubs.dtype)
  else:
    win = jnp.asarray(window)
    if len(win.shape) != 1:
      raise ValueError('window must be 1-D')
    if win.shape[0] != nperseg_int:
      raise ValueError(f'window must have length of {nperseg_int}')
  #xsubs *= win.sum()  # This takes care of the 'spectrum' scaling

  # make win broadcastable over xsubs
  win = lax.expand_dims(win, (*range(xsubs.ndim - 2), -1))
  x = _overlap_and_add((xsubs * win).swapaxes(-2, -1), nstep)
  #win_squared = jnp.repeat((win * win), xsubs.shape[-1], axis=-1)
  #norm = _overlap_and_add(win_squared.swapaxes(-2, -1), nstep)

  # Remove extension points
  if boundary:
    x = x[..., nperseg_int//2:-(nperseg_int//2)]
    norm = norm[..., nperseg_int//2:-(nperseg_int//2)]
  #x /= jnp.where(norm > 1e-10, norm, 1.0)

  # Put axes back
  if x.ndim > 1:
    if time_axis != Zxx.ndim - 1:
      if freq_axis < time_axis:
        time_axis -= 1
      x = jnp.moveaxis(x, -1, time_axis)

  time = jnp.arange(x.shape[0], dtype=np.finfo(x.dtype).dtype) / fs
  return time, x