"""
Gradient monitor: tracks gradient statistics (mean, std, min, max, norm,
pos/neg ratio, sign-flip ratio) at key layers during training.

Key monitored points (near-input → deep):
  - spectrogram  : Spectrogram extractor output grad
  - logmel       : LogmelFilterBank output grad
  - bn0          : BatchNorm2d after logmel (input-side gradient)
  - enc_conv1    : ResNet first conv weight grad
  - enc_layer1..4: ResNet stage output grads
  - fc           : Classification head weight grad
  - weight_base  : Learned base-class prototypes
  - weight_base_open : Learned open-set negative prototypes
  - pam          : PrototypeDynamicAggregation
  - ciam         : ConditionalInformationCoupling
  - npm          : OpenSetGenerater
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore', message='Full backward hook is firing when gradients are computed with respect to module outputs')


def _safe_stats(tensor):
    """Compute (mean, std, min, max, l2norm, pos_ratio, zero_ratio) of a tensor."""
    if tensor is None or tensor.numel() == 0:
        return None
    t = tensor.detach().float()
    return {
        'mean':     t.mean().item(),
        'std':      t.std().item(),
        'min':      t.min().item(),
        'max':      t.max().item(),
        'l2':       t.norm(p=2).item(),
        'pos_ratio':(t > 1e-8).float().mean().item(),
        'neg_ratio':(t < -1e-8).float().mean().item(),
        'zero_ratio':(t.abs() <= 1e-8).float().mean().item(),
        'numel':    t.numel(),
    }


def _sign_flip_ratio(cur, prev):
    """Ratio of elements whose sign flipped compared to previous step."""
    if cur is None or prev is None:
        return 0.0
    if cur.shape != prev.shape:
        return 0.0
    cur_s = cur.detach().float()
    prev_s = prev.detach().float()
    # Sign flip: (cur > 0 and prev < 0) or (cur < 0 and prev > 0)
    flipped = ((cur_s > 1e-8) & (prev_s < -1e-8)) | ((cur_s < -1e-8) & (prev_s > 1e-8))
    return flipped.float().mean().item()


class GradientMonitor:
    """Attaches backward hooks to key layers and logs gradient statistics.

    Usage:
        monitor = GradientMonitor(model, logger, writer)
        ... (hooks auto-register)
        # In training loop, after loss.backward():
        monitor.capture_param_grads()
        ... optimizer.step() ...
        monitor.log_step(iter_counter)
    """

    def __init__(self, model, logger, writer, enabled=True, log_interval=1):
        self.model = model
        self.logger = logger
        self.writer = writer
        self.enabled = enabled
        self.log_interval = log_interval
        self.step_count = 0

        # Buffers: per-layer_name -> tensor (latest grad_output or weight grad)
        self._grad_output = {}       # backward hook output grads
        self._grad_param = {}        # weight grads (after .backward())
        self._prev_grad_output = {}  # for sign-flip tracking
        self._prev_grad_param = {}

        # Hook handles
        self._handles = []

        if self.enabled and self.model is not None:
            self._register_hooks()

    # ---- hook registration ------------------------------------------------

    def _register_hooks(self):
        """Register backward hooks on key named modules."""
        targets = self._collect_targets()
        for name, module in targets:
            h = module.register_full_backward_hook(
                lambda m, g_in, g_out, n=name: self._backward_hook(n, g_out)
            )
            self._handles.append(h)

    def _collect_targets(self):
        """Walk the model and return [(name, module)] for key layers."""
        result = []
        # My_Net direct children
        for attr in ['spectrogram_extractor', 'logmel_extractor', 'bn0',
                     'PAM', 'CIAM', 'NPM', 'fc']:
            m = getattr(self.model, attr, None)
            if isinstance(m, nn.Module):
                result.append((attr, m))

        # Encoder internal layers (ResNet18)
        enc = getattr(self.model, 'encoder', None)
        if enc is not None:
            for attr in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']:
                m = getattr(enc, attr, None)
                if isinstance(m, nn.Module):
                    result.append((f'enc_{attr}', m))

        return result

    # ---- backward hook callback -------------------------------------------

    def _backward_hook(self, name, grad_output):
        """Called during backward(); captures grad_output[0] for output-side stats."""
        if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
            self._grad_output[name] = grad_output[0].detach().clone()

    # ---- parameter gradient capture (call after loss.backward()) ------------

    def capture_param_grads(self):
        """Capture .grad of key parameters after backward(). Call before optimizer.step()."""
        if not self.enabled:
            return

        for attr in ['weight_base', 'weight_base_open']:
            p = getattr(self.model, attr, None)
            if isinstance(p, nn.Parameter) and p.grad is not None:
                self._grad_param[attr] = p.grad.detach().clone()

        # fc weight grad
        fc = getattr(self.model, 'fc', None)
        if isinstance(fc, nn.Linear) and fc.weight.grad is not None:
            self._grad_param['fc_weight'] = fc.weight.grad.detach().clone()

        # encoder conv1 weight grad
        enc = getattr(self.model, 'encoder', None)
        if enc is not None:
            c1 = getattr(enc, 'conv1', None)
            if isinstance(c1, nn.Conv2d) and c1.weight.grad is not None:
                self._grad_param['enc_conv1_weight'] = c1.weight.grad.detach().clone()

    # ---- logging -----------------------------------------------------------

    def log_step(self, iter_counter):
        """Aggregate stats and log to TensorBoard + logger. Call after optimizer.step()."""
        if not self.enabled:
            return

        self.step_count += 1
        if self.step_count % self.log_interval != 0:
            self._swap_history()
            return

        # ---- output-side gradients (hooked from backward) ----
        for name, grad in self._grad_output.items():
            s = _safe_stats(grad)
            if s is None:
                continue
            flip = _sign_flip_ratio(grad, self._prev_grad_output.get(name))
            for k, v in s.items():
                self.writer.add_scalar(f'grad_output/{name}/{k}', v, iter_counter)
            self.writer.add_scalar(f'grad_output/{name}/sign_flip', flip, iter_counter)

        # ---- parameter gradients ----
        for name, grad in self._grad_param.items():
            s = _safe_stats(grad)
            if s is None:
                continue
            flip = _sign_flip_ratio(grad, self._prev_grad_param.get(name))
            for k, v in s.items():
                self.writer.add_scalar(f'grad_param/{name}/{k}', v, iter_counter)
            self.writer.add_scalar(f'grad_param/{name}/sign_flip', flip, iter_counter)

        # ---- near-input summary line to logger ----
        near_input_keys = ['spectrogram_extractor', 'logmel_extractor', 'bn0', 'enc_conv1']
        parts = []
        for k in near_input_keys:
            g = self._grad_output.get(k)
            if g is not None:
                parts.append(f"{k}: mean={g.mean().item():.2e} std={g.std().item():.2e} "
                             f"pos={_safe_stats(g)['pos_ratio']:.2f}")
        if parts:
            self.logger.info(f"[GradMonitor step={iter_counter}] near-input: " + " | ".join(parts))

        # ---- deep / task-module summary ----
        deep_keys = ['enc_layer4', 'PAM', 'CIAM', 'NPM', 'fc']
        parts2 = []
        for k in deep_keys:
            g = self._grad_output.get(k)
            if g is not None:
                parts2.append(f"{k}: mean={g.mean().item():.2e} std={g.std().item():.2e}")
        if parts2:
            self.logger.info(f"[GradMonitor step={iter_counter}] deep/task: " + " | ".join(parts2))

        # ---- param grad summary ----
        param_keys = ['weight_base', 'weight_base_open', 'fc_weight', 'enc_conv1_weight']
        parts3 = []
        for k in param_keys:
            g = self._grad_param.get(k)
            if g is not None:
                parts3.append(f"{k}: mean={g.mean().item():.2e} std={g.std().item():.2e}")
        if parts3:
            self.logger.info(f"[GradMonitor step={iter_counter}] param-grad: " + " | ".join(parts3))

        self._swap_history()

    def _swap_history(self):
        """Move current grads → prev for next sign-flip comparison."""
        self._prev_grad_output = {k: v.clone() for k, v in self._grad_output.items()}
        self._prev_grad_param = {k: v.clone() for k, v in self._grad_param.items()}
        self._grad_output.clear()
        self._grad_param.clear()

    def close(self):
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
