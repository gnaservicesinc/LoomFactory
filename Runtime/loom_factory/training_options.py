"""Validated training controls and optimizer-step schedules for fresh map models."""

from dataclasses import dataclass, asdict
import math

import torch


def _number(request, key, default, low, high, *, exclusive_low=False):
    value = request.get(key, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value > high or
            (value <= low if exclusive_low else value < low)):
        comparison = "greater than" if exclusive_low else "at least"
        raise ValueError(f"{key} must be finite, {comparison} {low} and at most {high}.")
    return float(value)


def _integer(request, key, default, low, high):
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}.")
    return value


@dataclass(frozen=True)
class TrainingOptions:
    optimizer: str
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    momentum: float
    gradient_accumulation_steps: int
    gradient_clip_norm: float
    lr_schedule: str
    warmup_epochs: int
    minimum_lr_ratio: float
    crop_refresh: str
    augmentation_refresh: str

    @classmethod
    def from_request(cls, request):
        optimizer = request.get("optimizer", "adamw")
        if optimizer not in ("adamw", "adam", "sgd", "lion"):
            raise ValueError("optimizer must be adamw, adam, sgd or lion.")
        schedule = request.get("lr_schedule", "constant")
        if schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be constant or cosine.")
        refresh = {}
        for key in ("crop_refresh", "augmentation_refresh"):
            refresh[key] = request.get(key, "per_epoch")
            if refresh[key] not in ("per_sample", "per_epoch", "per_run"):
                raise ValueError(f"{key} must be per_sample, per_epoch or per_run.")
        warmup = _integer(request, "warmup_epochs", 0, 0, 100000)
        epochs = _integer(request, "epochs", 10, 1, 100000)
        if warmup >= epochs:
            raise ValueError("Warmup epochs must be fewer than the total training epochs.")
        # Hidden settings from a previously selected optimizer do not affect the
        # active optimizer. Normalize them to finite defaults for provenance.
        beta_request = request if optimizer != "sgd" else {}
        beta1 = _number(beta_request, "beta1", .9, 0, 1)
        beta2 = _number(beta_request, "beta2", .99 if optimizer == "lion" else .999, 0, 1)
        if beta1 == 1 or beta2 == 1:
            raise ValueError("Optimizer beta values must be less than one.")
        return cls(
            optimizer=optimizer,
            learning_rate=_number(request, "learning_rate", .001, 0, 1, exclusive_low=True),
            weight_decay=_number(request, "weight_decay", .01, 0, 10),
            beta1=beta1, beta2=beta2,
            momentum=_number(request if optimizer == "sgd" else {}, "momentum", .9, 0, 1),
            gradient_accumulation_steps=_integer(request, "gradient_accumulation_steps", 1, 1, 4096),
            gradient_clip_norm=_number(request, "gradient_clip_norm", 1, 0, 1000000),
            lr_schedule=schedule, warmup_epochs=warmup,
            minimum_lr_ratio=_number(request, "minimum_lr_ratio", .1, 0, 1),
            **refresh,
        )

    def descriptor(self):
        return asdict(self)


class Lion(torch.optim.Optimizer):
    """Sign momentum optimizer from Chen et al., arXiv:2302.06675.

    Its update follows the published Google Research algorithm: the first beta
    blends the old momentum with the current gradient for the signed update;
    the second beta updates the stored momentum for the next optimizer step.
    """

    def __init__(self, params, lr, betas, weight_decay):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            first_beta, second_beta = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError("Lion requires dense model gradients.")
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter)
                momentum = state["momentum"]
                direction = momentum.mul(first_beta).add(gradient, alpha=1 - first_beta).sign_()
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(direction, alpha=-group["lr"])
                momentum.lerp_(gradient, 1 - second_beta)
        return loss


def make_optimizer(parameters, options):
    common = dict(lr=options.learning_rate, weight_decay=options.weight_decay)
    if options.optimizer == "sgd":
        return torch.optim.SGD(parameters, momentum=options.momentum, **common)
    betas = (options.beta1, options.beta2)
    if options.optimizer == "lion":
        return Lion(parameters, betas=betas, **common)
    optimizer = torch.optim.AdamW if options.optimizer == "adamw" else torch.optim.Adam
    return optimizer(parameters, betas=betas, **common)


def learning_rate_for_step(options, step, total_steps, steps_per_epoch):
    """LR for a zero-based optimizer step, including the final cosine endpoint."""
    warmup_steps = options.warmup_epochs * steps_per_epoch
    if step < warmup_steps:
        return options.learning_rate * (step + 1) / warmup_steps
    if options.lr_schedule == "constant":
        return options.learning_rate
    decay_steps = total_steps - warmup_steps
    fraction = (step - warmup_steps) / (decay_steps - 1) if decay_steps > 1 else 0.0
    ratio = options.minimum_lr_ratio + (1 - options.minimum_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2
    return options.learning_rate * ratio
