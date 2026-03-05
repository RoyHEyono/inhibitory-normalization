from inhib_norm.normalization import DivisiveNormalizeFunction, DivisiveNormalize
import torch
import unittest
import torch.nn as nn
from torch.autograd import gradcheck

class TestDivisiveNormalizeFunction(unittest.TestCase):

    class DivisiveNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-5):
            """
            Implements divisive normalization (only divides by std, does not subtract mean).
            
            Args:
                normalized_shape (int or tuple): Number of features (like LayerNorm).
                eps (float): Small constant for numerical stability.
            """
            super().__init__()
            self.eps = eps
            self.normalized_shape = normalized_shape

        def forward(self, x):
            # Compute variance (without mean subtraction)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + 1e-5)
            x_norm = x / std

            return x_norm

    def setUp(self):
        """Set up test tensors"""
        torch.manual_seed(42)
        self.batch_size = 4
        self.feature_dim = 4
        self.x = torch.randn(self.batch_size, self.feature_dim, requires_grad=True) - 1 # Subtract 1 to get it off center

    def test_divisive_norm_class(self):
        dim_x = 20
        x = torch.randn(32, dim_x)
        layernorm_control = nn.LayerNorm(dim_x, elementwise_affine=False)
        ln_local = self.DivisiveNorm(20)
        x_centered = x - x.mean(dim=1, keepdim=True)

        self.assertTrue(torch.allclose(layernorm_control(x_centered), ln_local(x_centered), atol=1e-8),
                        msg=f"Mismatch between divisive normalization and LayerNorm.")

    def test_forward_variance(self):
        """Check that DivisiveNormalizeFunction produces unit variance."""
        x_norm = DivisiveNormalizeFunction.apply(self.x, False)
        var = x_norm.var(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-4),
                        msg=f"Expected variance=1, got {var}")

    def test_forward_against_layernorm(self):
        """Compare output against LayerNorm but with mean fixed at 0"""
        ln = torch.nn.LayerNorm(self.feature_dim, elementwise_affine=False)
        out_x = self.x - self.x.mean(dim=-1, keepdim=True)

        ln_out = ln(out_x)

        dn_out = DivisiveNormalizeFunction.apply(out_x, False)

        self.assertTrue(torch.allclose(dn_out, ln_out, atol=1e-5),
                        msg=f"Mismatch between divisive normalization and LayerNorm.\n{dn_out - ln_out}")

    class LayerNormNet(nn.Module):
            """A simple network that applies a linear transformation followed by Layer Norm."""
            def __init__(self, feature_dim, divisive_norm_func, use_custom_norm=False,  dtype=torch.float64):
                super().__init__()
                self.linear = nn.Linear(feature_dim, feature_dim, bias=False, dtype=dtype)  # ✅ Init directly in float64
                self.norm = (
                    DivisiveNormalize(no_backward=False, no_forward=False).to(dtype=dtype)
                    if use_custom_norm else divisive_norm_func  # ✅ Directly in float64
                )

            def forward(self, x):
                return self.norm(self.linear(x))

    def test_backward_against_layernorm(self):
        """Compare gradients of LayerNormalizeFunction vs. PyTorch's LayerNorm using autograd checks."""
        feature_dim, batch_size = 10, 5
        norm_func = self.DivisiveNorm(feature_dim)

        # ✅ Ensure input is structured and meaningful
        x = torch.arange(batch_size * feature_dim, dtype=torch.float64).view(batch_size, feature_dim)
        x.requires_grad_(True)

        # ✅ Create first model (PyTorch LayerNorm)
        model_torch = self.LayerNormNet(feature_dim, norm_func, use_custom_norm=False, dtype=torch.float64)

        # ✅ Create second model (Custom LayerNorm) with **identical weights**
        model_custom = self.LayerNormNet(feature_dim, norm_func, use_custom_norm=True, dtype=torch.float64)
        model_custom.linear.weight.data.copy_(model_torch.linear.weight.data)  # ✅ Copy identical weights

        # Compute scalar loss (sum ensures gradients exist)
        loss_torch = model_torch(x).sum()
        loss_custom = model_custom(x).sum()

        # Compute gradients w.r.t. x
        grad_x_torch, = torch.autograd.grad(loss_torch, x, create_graph=True)
        grad_x_custom, = torch.autograd.grad(loss_custom, x, create_graph=True)

        # ✅ Compare outputs and gradients
        self.assertTrue(torch.allclose(loss_torch, loss_custom, atol=1e-25),
                        msg=f"Output mismatch: {loss_torch.item()} vs {loss_custom.item()}")

        self.assertTrue(torch.allclose(grad_x_torch, grad_x_custom, atol=1e-8),
                        msg=f"Gradient mismatch.\n{grad_x_torch - grad_x_custom}")

        # ✅ Check parameter gradients
        model_torch.zero_grad()
        model_custom.zero_grad()
        loss_torch.backward(retain_graph=True)
        loss_custom.backward()
        self.assertTrue(torch.allclose(model_torch.linear.weight.grad, model_custom.linear.weight.grad, atol=1e-8),
                        msg="Linear weight gradients mismatch.")

        # ✅ Use gradcheck on the **custom model**, ensuring everything is float64
        x_gradcheck = x.detach().clone().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(model_custom, (x_gradcheck,), eps=1e-6, atol=1e-4))

        




if __name__ == "__main__":
    unittest.main()
