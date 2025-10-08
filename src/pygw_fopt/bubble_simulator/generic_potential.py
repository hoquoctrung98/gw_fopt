#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Quoc Trung Ho <qho@sissa.it>
"""

import numpy as np

from typing import Any
from cosmoTransitions.generic_potential import generic_potential

# working
class GenericPotential(generic_potential):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Base class for a general potential function in multiple variables.
        """
        generic_potential.__init__(self, *args, **kwargs)
        self.params_str = None

    def V0(self, X: np.ndarray) -> float:
        """
        Compute the potential at tree level V0 at given points.
        
        Parameters:
            X (np.ndarray): A NumPy array of shape (..., Ndim), where the last axis represents the field variables.
        
        Returns:
            float: The potential value at X.
        
        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The potential function V0 must be defined in a subclass.")

    def dV0(self, X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Compute the gradient of the potential V0 numerically using central finite differences.
        If a subclass defines an explicit dV method, it will override this behavior.
        
        Parameters:
            X (np.ndarray): A NumPy array of shape (..., Ndim) representing input points.
            eps (float): Step size for finite difference approximation.
        
        Returns:
            np.ndarray: A NumPy array of the same shape as X, containing the gradient of V0.
        """
        X = np.asarray(X, dtype=float)
        grad = np.zeros_like(X)
        for i in range(X.shape[-1]):
            dX = np.zeros_like(X)
            dX[..., i] = eps  # Perturb only the i-th field component
            
            V_plus = self.V0(X + dX)
            V_minus = self.V0(X - dX)
            grad[..., i] = (V_plus - V_minus) / (2 * eps)  # Central difference
        return grad

    def d2V0(self, X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Compute the Hessian matrix (second derivatives) of the potential V0 numerically.
        If a subclass defines an explicit d2V method, it will override this behavior.
        
        Parameters:
            X (np.ndarray): A NumPy array of shape (..., Ndim) representing input points.
            eps (float): Step size for finite difference approximation.
        
        Returns:
            np.ndarray: A NumPy array of shape (Ndim, Ndim) representing the Hessian matrix.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[-1]
        hessian = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                dX_i = np.zeros_like(X)
                dX_j = np.zeros_like(X)
                dX_i[..., i] = eps
                dX_j[..., j] = eps
                if i == j:
                    # Second derivative w.r.t. the same variable
                    V_pp = self.V0(X + dX_i)
                    V_mm = self.V0(X - dX_i)
                    V_0 = self.V0(X)
                    hessian[i, j] = (V_pp - 2 * V_0 + V_mm) / (eps**2)
                else:
                    # Mixed partial derivative
                    V_pq = self.V0(X + dX_i + dX_j)
                    V_p = self.V0(X + dX_i - dX_j)
                    V_q = self.V0(X - dX_i + dX_j)
                    V_mq = self.V0(X - dX_i - dX_j)
                    hessian[i, j] = (V_pq - V_p - V_q + V_mq) / (4 * eps**2)
        return hessian