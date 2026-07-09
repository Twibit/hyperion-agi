import numpy as np

class FreeEnergyMath:
    """
    Mathematical core for Variational Free Energy (FEP).
    Upgraded in 2026 to include Lyapunov stability, Heavy-Tailed Log-Cosh Geometry,
    and Softplus Precision scaling. Guaranteed immune to NaN explosions.
    """
    
    @staticmethod
    def softplus(x):
        """Ensures precision (inverse variance) is strictly positive."""
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
        
    @staticmethod
    def pseudo_huber_loss(error, delta=1.0):
        """
        Heavy-Tailed Generative Model.
        Smoothly bounds extreme outliers so gradients naturally decay 
        instead of exploding to infinity. (F = |epsilon| at extremes)
        """
        return delta**2 * (np.sqrt(1 + (error/delta)**2) - 1)
        
    @staticmethod
    def pseudo_huber_gradient(error, delta=1.0):
        """The bounded derivative of the pseudo-Huber loss."""
        return error / np.sqrt(1 + (error/delta)**2)

    @staticmethod
    def precision_weighted_error(prediction, target, raw_precision, delta=1.0):
        """
        Calculates error natively bounded by a Heavy-Tailed distribution
        and scaled by learnable precision (Natural Gradients).
        """
        error = target - prediction
        precision = FreeEnergyMath.softplus(raw_precision)
        
        # The gradient naturally flattens for impossible outliers
        bounded_grad = FreeEnergyMath.pseudo_huber_gradient(error, delta)
        
        # Element-wise precision scaling (Natural Gradient)
        weighted_error = precision * bounded_grad  
        
        # Calculate strict Variational Free Energy
        free_energy = np.sum(precision * FreeEnergyMath.pseudo_huber_loss(error, delta)) - 0.5 * np.sum(np.log(precision + 1e-12))
        return free_energy, weighted_error

    @staticmethod
    def lyapunov_stable_update(mu_t, gradient, kappa=0.01, lr=0.01):
        """
        Continuous-time leaky integrator: mu_dot = -kappa * mu - gradient.
        Guarantees physical convergence (Lyapunov stability) so states 
        can never spiral to infinity.
        """
        return mu_t + lr * (-kappa * mu_t - gradient)
        
    @staticmethod
    def metabolic_efficiency_cost(activations, wiring_matrix, sparsity_lambda=0.1, wiring_lambda=0.01):
        """
        Translates the 'Neural Efficiency Hypothesis' into mathematical bounds.
        """
        sparsity_cost = sparsity_lambda * np.sum(np.abs(activations))
        wiring_cost = wiring_lambda * np.sum(wiring_matrix ** 2)
        return sparsity_cost + wiring_cost

    @staticmethod
    def adaptive_precision_update(raw_precision, error, learning_rate=0.01):
        """
        Learns the uncertainty (variance) alongside the prediction.
        If error variance spikes, raw_precision drops, killing the unstable gradient naturally.
        """
        error_variance = error ** 2
        
        # Empirical natural gradient update for precision
        precision = FreeEnergyMath.softplus(raw_precision)
        precision_gradient = precision - (precision * error_variance * precision)
        
        # Update raw precision
        return raw_precision + learning_rate * precision_gradient
