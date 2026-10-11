"""Latent shards: the AudioVAE latents of many clips in one file, addressed by uid.

A shard is a torch file {"uids": [str], "offsets": int64 [N + 1], "latents": float16 [frames, 64]}; clip i is
latents[offsets[i]:offsets[i + 1]].
"""

from __future__ import annotations

import os

import torch


class ShardWriter:
    def __init__(self, path: str):
        self.path, self.uids, self.parts = path, [], []

    def add(self, uid: str, latents: torch.Tensor) -> None:
        self.uids.append(uid)
        self.parts.append(latents.to("cpu", torch.float16))

    def close(self) -> None:
        offsets = torch.zeros(len(self.parts) + 1, dtype=torch.int64)
        if self.parts:
            offsets[1:] = torch.cumsum(torch.tensor([len(p) for p in self.parts]), 0)
        latents = torch.cat(self.parts) if self.parts else torch.zeros(0, 64, dtype=torch.float16)
        tmp = self.path + ".tmp"
        torch.save({"uids": self.uids, "offsets": offsets, "latents": latents}, tmp)
        os.replace(tmp, self.path)  # a shard exists only once complete


class Shard:
    def __init__(self, path: str):
        d = torch.load(path, map_location="cpu", weights_only=True)
        self.offsets, self.latents = d["offsets"], d["latents"]
        self.index = {u: i for i, u in enumerate(d["uids"])}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, uid: str) -> torch.Tensor:
        i = self.index[uid]
        return self.latents[self.offsets[i] : self.offsets[i + 1]]
