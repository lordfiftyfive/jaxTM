from tmu.tmulib import ffi, lib
from tmu.tools import CFFISerializable
import numpy as np

import jax.numpy as jnp


class WeightBank(CFFISerializable):

    def __init__(self, weights: np.ndarray, copy: bool = True):
        self.number_of_clauses = weights.shape[0]
        self.weights = weights.copy(order="C") if copy else weights
        
        
        self._cffi_init()

    def _cffi_init(self):
        self.cw_p = ffi.cast("int *", self.weights.ctypes.data)

    def increment(self, clause_output, update_p, clause_active, positive_weights):
        co_p = ffi.cast("unsigned int *", clause_output.ctypes.data)
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.wb_increment(self.cw_p, self.number_of_clauses, co_p, update_p, ca_p, int(positive_weights))

    def decrement(self, clause_output, update_p, clause_active, negative_weights):
        co_p = ffi.cast("unsigned int *", clause_output.ctypes.data)
        ca_p = ffi.cast("unsigned int *", clause_active.ctypes.data)
        lib.wb_decrement(self.cw_p, self.number_of_clauses, co_p, update_p, ca_p, int(negative_weights))

    def get_weights(self):
        return self.weights
