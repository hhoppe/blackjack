# Adaptation of https://github.com/numba/numba/blob/main/numba/cuda/random.py to:
# - Use four uint32 instead of two uint64.
# - Allow access to uint32.
# - Allow rng state to be stored in four registers.

# See http://prng.di.unimi.it/xoshiro128plus.c  -- xoshiro128+.
# 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
# "xoshiro128+ 1.0, our best and fastest 32-bit generator for 32-bit floating-point numbers".

import math
from typing import Any

from numba import (config, cuda, float32, uint32, int32, from_dtype, jit)
import numpy as np

_CudaArray = Any  # cuda.cudadrv.devicearray.DeviceNDArray

# Untyped decorator makes function untyped.
# mypy: disable-error-code="misc"

# This implementation is based upon the xoshiro128+ algorithm described at:
#     http://xoroshiro.di.unimi.it/
#
# Originally implemented by David Blackman and Sebastiano Vigna.
# This is a 32-bit version adapted from the 64-bit implementation.

# Configuration for object mode when using CUDA simulator.
_forceobj = _looplift = config.ENABLE_CUDASIM  # pylint: disable=no-member
_nopython = not config.ENABLE_CUDASIM  # pylint: disable=no-member

# Define the state dtype for xoshiro128+.
xoshiro128p_dtype = np.dtype(
    [('s0', np.uint32), ('s1', np.uint32), ('s2', np.uint32), ('s3', np.uint32)], align=True
)
xoshiro128p_type = from_dtype(xoshiro128p_dtype)


@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def rotl(x: uint32, k: uint32) -> uint32:
  """Left rotate x by k bits."""
  x = uint32(x)
  k = uint32(k)
  return (x << k) | (x >> uint32(32 - k))


@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def init_xoshiro128p_state(states: _CudaArray, index: int32, seed: uint32) -> None:
  """Initialize xoshiro128+ state from a 32-bit seed using SplitMix32.

  Args:
    states: 1D array with dtype=xoshiro128p_dtype that holds RNG states.
    index: Offset in states array to update.
    seed: Initial seed value for state initialization.
  """
  index = int32(index)
  seed = uint32(seed)

  # Simple 32-bit mixing function for initialization.
  z = seed
  z = uint32((z ^ (z >> uint32(16))) * uint32(0x85EBCA6B))
  z = uint32((z ^ (z >> uint32(13))) * uint32(0xC2B2AE35))
  z = uint32(z ^ (z >> uint32(16)))

  states[index]['s0'] = z
  z = uint32(z + uint32(0x9E3779B9))
  states[index]['s1'] = z
  z = uint32(z + uint32(0x9E3779B9))
  states[index]['s2'] = z
  z = uint32(z + uint32(0x9E3779B9))
  states[index]['s3'] = z


@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def xoshiro128p_next(states: _CudaArray, index: int32) -> uint32:
  """Return the next random uint32 and advance the RNG in states[index].

  Args:
    states: 1D array with dtype=xoshiro128p_dtype that holds RNG states.
    index: Offset in states array to update.

  Returns:
    A uint32 random number from the sequence.
  """
  index = int32(index)

  result = uint32(states[index]['s0'] + states[index]['s3'])
  t = uint32(states[index]['s1'] << uint32(9))

  states[index]['s2'] ^= states[index]['s0']
  states[index]['s3'] ^= states[index]['s1']
  states[index]['s1'] ^= states[index]['s2']
  states[index]['s0'] ^= states[index]['s3']

  states[index]['s2'] ^= t
  states[index]['s3'] = rotl(states[index]['s3'], uint32(11))

  return result


@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def xoshiro128p_next_raw(
    s0: uint32, s1: uint32, s2: uint32, s3: uint32
) -> tuple[uint32, uint32, uint32, uint32, uint32]:
  """Return the next random uint32 and updated state values.

  Args:
    s0, s1, s2, s3: Current state values in registers.

  Returns:
    Tuple of (random_value, new_s0, new_s1, new_s2, new_s3).
  """
  result = uint32(s0 + s3)
  t = uint32(s1 << 9)

  s2 ^= s0
  s3 ^= s1
  s1 ^= s2
  s0 ^= s3

  s2 ^= t
  s3 = rotl(s3, 11)

  return result, s0, s1, s2, s3


@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def xoshiro128p_jump(states: _CudaArray, index: int32) -> None:
  """Advance the RNG in states[index] by 2^64 steps.

  Args:
    states: 1D array with dtype=xoshiro128p_dtype that holds RNG states.
    index: Offset in states array to update.
  """
  index = int32(index)

  JUMP = (uint32(0x8764000B), uint32(0xF542D2D3), uint32(0x6FA035C3), uint32(0x77F2DB5B))

  s0 = uint32(0)
  s1 = uint32(0)
  s2 = uint32(0)
  s3 = uint32(0)

  for i in range(4):
    for b in range(32):
      if JUMP[i] & (uint32(1) << uint32(b)):
        s0 ^= states[index]['s0']
        s1 ^= states[index]['s1']
        s2 ^= states[index]['s2']
        s3 ^= states[index]['s3']
      xoshiro128p_next(states, index)

  states[index]['s0'] = s0
  states[index]['s1'] = s1
  states[index]['s2'] = s2
  states[index]['s3'] = s3


@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def uint32_to_unit_float32(x: uint32) -> float32:
  """Convert uint32 to float32 value in the range [0.0, 1.0)."""
  x = uint32(x)
  return float32(x >> uint32(8)) * float32(1.0 / (1 << 24))


@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def xoshiro128p_uniform_float32(states: _CudaArray, index: int32) -> float32:
  """Return a float32 in range [0.0, 1.0) and advance states[index].

  Args:
    states: 1D array with dtype=xoshiro128p_dtype that holds RNG states.
    index: Offset in states array to update.

  Returns:
    A float32 random number uniformly distributed in [0.0, 1.0).
  """
  index = int32(index)
  return uint32_to_unit_float32(xoshiro128p_next(states, index))


TWO_PI_FLOAT32 = float32(2 * math.pi)


@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def xoshiro128p_normal_float32(states: _CudaArray, index: int32) -> float32:
  """Return a normally distributed float32 and advance states[index].

  The return value is drawn from a Gaussian of mean=0 and sigma=1 using the
  Box-Muller transform. This advances the RNG sequence by two steps.

  :type states: 1D array, dtype=xoshiro128p_dtype
  :param states: array of RNG states
  :type index: int32
  :param index: offset in states to update
  :rtype: float32
  """
  index = int32(index)

  u1 = xoshiro128p_uniform_float32(states, index)
  u2 = xoshiro128p_uniform_float32(states, index)

  z0 = math.sqrt(float32(-2.0) * math.log(u1)) * math.cos(TWO_PI_FLOAT32 * u2)
  return z0


@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def init_xoshiro128p_states_cpu(
    states: _CudaArray, seed: uint32, subsequence_start: uint32
) -> None:
  """Initialize RNG states on CPU for parallel generators."""
  n = states.shape[0]
  seed = uint32(seed)
  subsequence_start = uint32(subsequence_start)

  if n >= 1:
    init_xoshiro128p_state(states, 0, seed)

    # Advance to starting subsequence number.
    for _ in range(subsequence_start):
      xoshiro128p_jump(states, 0)

    # Populate the rest of the array.
    for i in range(1, n):
      states[i] = states[i - 1]  # Take state of previous generator.
      xoshiro128p_jump(states, i)  # Jump forward 2^64 steps.


def init_xoshiro128p_states(
    states: _CudaArray, seed: uint32, subsequence_start: uint32 = 0, stream: int = 0
) -> None:
  """Initialize RNG states on the GPU for parallel generators.

  This initializes the RNG states so that each state in the array corresponds to subsequences separated
  by 2^64 steps from each other in the main sequence. Therefore, as long no CUDA thread requests more
  than 2^64 random numbers, all of the RNG states produced by this function are guaranteed to be
  independent.

  Args:
    states: DeviceNDArray with dtype=xoshiro128p_dtype that holds RNG states.
    seed: Starting seed for list of generators.
    subsequence_start: Number of 2^64 steps to advance first state.
    stream: CUDA stream to run initialization kernel on.
  """
  # Initialization on CPU is much faster than the GPU.
  states_cpu = np.empty(shape=states.shape, dtype=xoshiro128p_dtype)
  init_xoshiro128p_states_cpu(states_cpu, seed, subsequence_start)
  states.copy_to_device(states_cpu, stream=stream)


def create_xoshiro128p_states(
    n: int, seed: uint32, subsequence_start: uint32 = 0, stream: int = 0
) -> _CudaArray:
  """Return a new device array initialized for n random number generators.

  This initializes the RNG states so that each state in the array corresponds to subsequences separated
  by 2^64 steps from each other in the main sequence. Therefore, as long no CUDA thread requests more
  than 2^64 random numbers, all of the RNG states produced by this function are guaranteed to be
  independent.

  Args:
    n: Number of RNG states to create.
    seed: Starting seed for list of generators.
    subsequence_start: Number of 2^64 steps to advance first state.
    stream: CUDA stream to run initialization kernel on.
  """
  states = cuda.device_array(n, dtype=xoshiro128p_dtype, stream=stream)
  init_xoshiro128p_states(states, seed, subsequence_start, stream)
  return states
