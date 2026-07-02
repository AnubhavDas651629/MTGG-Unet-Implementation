"""
Trainer for UNet-MTGNN Tornado Forecasting.

Handles training, validation, and loss computation for the hybrid model
that predicts spatial tornado probability maps from temporal weather sequences.

Loss: Weighted BCEWithLogitsLoss (same as the original Weather-Forecasting-Unet project)
Validation metric: KL divergence between predicted and hindcast probabilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


def bernoulli_entropy(p, eps=1e-7):
    """H(p) = -p*log(p) - (1-p)*log(1-p), safe for p in {0, 1}."""
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def kl_divergence_from_logits(logits, target, eps=1e-7):
    """
    Numerically stable KL(target || pred) for Bernoulli distributions.
    Uses the identity: KL(p || q) = BCE(p, q) - H(p)
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    h_target = bernoulli_entropy(target, eps).mean()
    return bce - h_target


class Trainer:
    """
    Training engine for the UNet-MTGNN hybrid model.

    Args:
        model:      UNetMTGNN model instance
        lrate:      Learning rate for Adam optimizer
        wdecay:     Weight decay (L2 regularization)
        clip:       Gradient clipping max norm (None to disable)
        device:     'cpu' or 'cuda'
        pos_weight: Positive class weight for BCEWithLogitsLoss (tornado events
                    are rare, so we upweight them). Default 10.0.
    """

    def __init__(self, model, lrate, wdecay, clip, device, pos_weight=10.0):
        self.model = model
        self.model.to(device)
        self.device = device
        self.clip = clip

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lrate, weight_decay=wdecay
        )

        # Weighted BCE loss — tornado events are extremely rare (class imbalance)
        # pos_weight > 1 makes the model pay more attention to tornado-positive pixels
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )

    def train(self, input_seq, target_seq):
        """
        Run one training step.

        Args:
            input_seq:  (B, seq_in, 3, H, W) — input weather map sequence
            target_seq: (B, seq_out, 2, H, W) — target tornado probability maps

        Returns:
            (loss, kl_div) — training loss and KL divergence for this batch
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass: model outputs raw logits (B, seq_out, 2, H, W)
        logits = self.model(input_seq)

        # Compute BCE loss
        loss = self.loss_fn(logits, target_seq)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        # Compute KL divergence for monitoring (detached, no gradients)
        with torch.no_grad():
            kl = kl_divergence_from_logits(logits, target_seq).item()

        return loss.item(), kl

    def eval(self, input_seq, target_seq):
        """
        Run one evaluation step (no gradient computation).

        Args:
            input_seq:  (B, seq_in, 3, H, W)
            target_seq: (B, seq_out, 2, H, W)

        Returns:
            (loss, kl_div) — validation loss and KL divergence for this batch
        """
        self.model.eval()

        with torch.no_grad():
            logits = self.model(input_seq)
            loss = self.loss_fn(logits, target_seq)
            kl = kl_divergence_from_logits(logits, target_seq).item()

        return loss.item(), kl


class Optim(object):
    """Generic optimizer wrapper with learning rate decay."""

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        self.optimizer.step()
        return  grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()
