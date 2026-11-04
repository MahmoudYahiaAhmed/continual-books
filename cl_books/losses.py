"""Loss pieces kept free of model code so they can be unit-tested on CPU."""
import torch
import torch.nn.functional as F


def topk_distill_loss(
    student_logits: torch.Tensor, topk_ids: torch.Tensor, topk_logprobs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy from the teacher's renormalised top-k distribution to the student.

    student_logits: [N, V] at the answer positions; topk_ids / topk_logprobs: [N, k]; mask: [N] (1 = real position).
    Equal to KL(teacher_topk || student) up to the teacher's constant entropy.
    """
    student_lp = F.log_softmax(student_logits.float(), dim=-1).gather(-1, topk_ids)
    teacher_p = F.softmax(topk_logprobs.float(), dim=-1)
    per_position = -(teacher_p * student_lp).sum(-1)
    return (per_position * mask).sum() / mask.sum().clamp(min=1)
