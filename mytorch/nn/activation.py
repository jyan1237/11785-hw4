import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # Implement forward pass
        max_of_each_row = Z.max(axis=self.dim, keepdims=True)
        adjust_Z = Z - max_of_each_row
        self.A = np.exp(adjust_Z) / np.sum(np.exp(adjust_Z), axis=self.dim, keepdims=True)
        return self.A  

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Implement backward pass
        # convert dLdA into (N, C) format from arbitrary input format
        dLdA = np.swapaxes(dLdA, self.dim, -1)
        dLdA = dLdA.reshape(-1, dLdA.shape[-1])
        A_swap = np.swapaxes(self.A, self.dim, -1)
        A_reshape = A_swap.reshape(-1, A_swap.shape[-1])

        N = dLdA.shape[0] 
        C = dLdA.shape[1]

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N,C))

        # Fill dLdZ one data point (row) at a time.
        for i in range(N):
            # Initialize the Jacobian with all zeros.
            # Hint: Jacobian matrix for softmax is a _×_ matrix, but what is _ here?
            J = np.zeros((C,C))

            # Fill the Jacobian matrix, please read the writeup for the conditions.
            for m in range(C):
                for n in range(C):
                    J[m, n] = A_reshape[i, m] * (1 - A_reshape[i, m]) if m==n else -A_reshape[i, m]*A_reshape[i, n]

            # Calculate the derivative of the loss with respect to the i-th input, please read the writeup for it.
            # Hint: How can we use (1×C) and (C×C) to get (1×C) and stack up vertically to give (N×C) derivative matrix?
            dLdZ[i, :] = dLdA[i] @ J
        
        dLdZ = dLdZ.reshape(A_swap.shape)
        dLdZ = np.swapaxes(dLdZ, self.dim, -1)

        return dLdZ
 

    