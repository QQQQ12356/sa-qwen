"""Optional question-conditioned Beacon writer; no document state is persisted."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionMemoryMixin:
    def init_question_memory(self, hidden_size, rank):
        self.question_down = nn.Linear(hidden_size, rank, bias=False)
        self.question_up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.question_up.weight)
        self.importance = nn.Linear(rank * 2, self.num_heads)

    def forward(self, beacons, previous, question):
        question_features = F.silu(self.question_down(self.norm(question)))
        condition = self.question_up(question_features.mean(dim=1, keepdim=True))
        features = F.silu(self.down(self.norm(beacons + condition)))
        projected = self.up(features)
        key, value, decay, beta = projected.split(
            [self.num_heads * self.key_dim, self.num_heads * self.value_dim,
             self.num_heads, self.num_heads], dim=-1,
        )
        batch, count = beacons.shape[:2]
        key = F.normalize(key.reshape(batch, count, self.num_heads, self.key_dim).float(), dim=-1)
        value = value.reshape(batch, count, self.num_heads, self.value_dim).float()
        pooled_question = question_features.mean(dim=1, keepdim=True).expand(-1, count, -1)
        relevance = self.importance(torch.cat([features, pooled_question], dim=-1)).float().sigmoid()
        state = value.new_zeros(batch, self.num_heads, self.key_dim, self.value_dim)
        if previous is not None:
            state = previous.float().clone()
        for token_idx in range(count):
            token_key = key[:, token_idx]
            prediction = (state * token_key.unsqueeze(-1)).sum(-2)
            residual = value[:, token_idx] - prediction
            residual_energy = residual.square().mean(-1)
            scale = value[:, token_idx].square().mean(-1) + prediction.square().mean(-1)
            novelty = residual_energy / (residual_energy + scale + 1e-6)
            importance = relevance[:, token_idx] * novelty
            retention = torch.exp(-importance * F.softplus(-decay[:, token_idx].float()) / count)
            state = state * retention[..., None, None]
            prediction = (state * token_key.unsqueeze(-1)).sum(-2)
            update = importance.unsqueeze(-1) * beta[:, token_idx].float().sigmoid().unsqueeze(-1)
            update = update * (value[:, token_idx] - prediction)
            state = state + token_key.unsqueeze(-1) * update.unsqueeze(-2)
        return state

    def readout_loss(self, student, teacher, question):
        with torch.no_grad():
            projected = self.up(F.silu(self.down(self.norm(question.detach()))))
            probes = projected[..., :self.num_heads * self.key_dim]
            probes = F.normalize(
                probes.reshape(question.shape[0], -1, self.num_heads, self.key_dim).float(), dim=-1,
            )
            target = torch.einsum("bthk,bhkv->bthv", probes, teacher.detach().float())
        prediction = torch.einsum("bthk,bhkv->bthv", probes, student)
        return (prediction - target).square().mean() / target.square().mean().clamp_min(1e-4)
