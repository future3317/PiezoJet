"""Small, model-agnostic gradient diagnostics.

These helpers are retained for auditing optimization interactions.  They do
not define a PiezoJet training protocol and are not imported by the production
trainer.
"""

from __future__ import annotations

from typing import Iterable

import torch


def factor_protected_gradient_projection(
    factor_loss: torch.Tensor,
    response_loss: torch.Tensor,
    protected_parameters: Iterable[torch.nn.Parameter],
    all_parameters: Iterable[torch.nn.Parameter],
    norm_match: bool = False,
) -> dict[str, float | int | bool]:
    """Write a one-sided factor-protected gradient for a diagnostic update."""
    protected = [parameter for parameter in protected_parameters if parameter.requires_grad]
    protected_ids = {id(parameter) for parameter in protected}
    all_trainable = [parameter for parameter in all_parameters if parameter.requires_grad]
    unprotected = [parameter for parameter in all_trainable if id(parameter) not in protected_ids]
    if not protected:
        raise ValueError("Factor-protected projection requires a non-empty trainable factor stack")

    factor_grads = torch.autograd.grad(factor_loss, protected, retain_graph=True, allow_unused=True)
    response_grads = torch.autograd.grad(response_loss, protected, retain_graph=True, allow_unused=True)
    response_only_grads = (
        torch.autograd.grad(response_loss, unprotected, retain_graph=False, allow_unused=True)
        if unprotected
        else ()
    )
    factor_parts = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(protected, factor_grads)
    ]
    response_parts = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(protected, response_grads)
    ]
    factor_vector, response_vector = torch.cat(factor_parts), torch.cat(response_parts)
    factor_norm = torch.linalg.vector_norm(factor_vector)
    response_norm = torch.linalg.vector_norm(response_vector)
    epsilon = torch.finfo(factor_vector.dtype).eps
    dot = torch.dot(factor_vector, response_vector)
    conflict = bool((dot < 0).detach()) and bool((factor_norm > epsilon).detach())
    coefficient = dot / factor_norm.square().clamp_min(epsilon) if conflict else torch.zeros_like(dot)
    projected_response_grads = [
        None
        if response_gradient is None and factor_gradient is None
        else (
            (torch.zeros_like(parameter) if response_gradient is None else response_gradient)
            - coefficient * (torch.zeros_like(parameter) if factor_gradient is None else factor_gradient)
        )
        for parameter, factor_gradient, response_gradient in zip(protected, factor_grads, response_grads)
    ]
    projected_response_vector = torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
            for parameter, gradient in zip(protected, projected_response_grads)
        ]
    )
    projected_response_norm = torch.linalg.vector_norm(projected_response_vector)
    response_scale = (
        factor_norm / projected_response_norm.clamp_min(epsilon)
        if norm_match and bool((projected_response_norm > epsilon).detach())
        else torch.ones_like(factor_norm)
    )
    for parameter, factor_gradient, response_gradient in zip(
        protected, factor_grads, projected_response_grads
    ):
        factor_value = torch.zeros_like(parameter) if factor_gradient is None else factor_gradient
        response_value = torch.zeros_like(parameter) if response_gradient is None else response_gradient
        parameter.grad = (factor_value + response_scale * response_value).detach()
    for parameter, gradient in zip(unprotected, response_only_grads):
        parameter.grad = (torch.zeros_like(parameter) if gradient is None else gradient).detach()

    cosine = dot / (factor_norm * response_norm).clamp_min(epsilon)
    removed = torch.linalg.vector_norm(response_vector - projected_response_vector)
    return {
        "factor_protected_parameters": sum(parameter.numel() for parameter in protected),
        "factor_gradient_norm": float(factor_norm.detach()),
        "response_gradient_norm": float(response_norm.detach()),
        "factor_response_gradient_cosine_before_projection": float(cosine.detach()),
        "response_gradient_norm_after_projection": float(projected_response_norm.detach()),
        "response_gradient_scale_after_projection": float(response_scale.detach()),
        "factor_protected_norm_matching": norm_match,
        "removed_response_gradient_fraction": float(
            (removed / response_norm.clamp_min(epsilon)).detach()
        ),
        "response_gradient_conflict_projected": conflict,
    }


def gradient_conflict_metrics(
    factor_loss: torch.Tensor,
    response_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
) -> dict[str, float | int | None]:
    """Measure two task gradients without mutating ``parameter.grad``."""
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        return {
            "shared_encoder_trainable_parameters": 0,
            "lambda_gradient_norm": None,
            "piezo_gradient_norm": None,
            "gradient_cosine": None,
        }
    factor_grads = torch.autograd.grad(factor_loss, trainable, retain_graph=True, allow_unused=True)
    response_grads = torch.autograd.grad(response_loss, trainable, retain_graph=True, allow_unused=True)
    factor_parts, response_parts = [], []
    for factor_grad, response_grad in zip(factor_grads, response_grads):
        if factor_grad is None and response_grad is None:
            continue
        factor_parts.append(
            torch.zeros_like(response_grad).reshape(-1)
            if factor_grad is None
            else factor_grad.reshape(-1)
        )
        response_parts.append(
            torch.zeros_like(factor_grad).reshape(-1)
            if response_grad is None
            else response_grad.reshape(-1)
        )
    if not factor_parts:
        return {
            "shared_encoder_trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "lambda_gradient_norm": 0.0,
            "piezo_gradient_norm": 0.0,
            "gradient_cosine": 0.0,
        }
    factor_vector, response_vector = torch.cat(factor_parts), torch.cat(response_parts)
    factor_norm = torch.linalg.vector_norm(factor_vector)
    response_norm = torch.linalg.vector_norm(response_vector)
    cosine = torch.dot(factor_vector, response_vector) / (
        factor_norm * response_norm
    ).clamp_min(torch.finfo(factor_vector.dtype).eps)
    return {
        "shared_encoder_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "lambda_gradient_norm": float(factor_norm.detach()),
        "piezo_gradient_norm": float(response_norm.detach()),
        "gradient_cosine": float(cosine.detach()),
    }
