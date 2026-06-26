"""
gaussian_lifecycle.py — Dynamic Gaussian Lifecycle Manager

Tracks per-object Gaussian groups through time. Objects that stop appearing
have their Gaussians gradually faded out (opacity reduced over FADE_STEPS
frames) rather than deleted. Objects that re-appear have opacity restored.

Confidence dynamics:
  - Seen this frame:      confidence += RESTORE_RATE, capped at 1.0
  - Not seen this frame:  confidence -= DECAY_RATE, floored at 0.0
  - confidence < STALE_THRESHOLD → start fade sequence
  - Object re-detected mid-fade → opacity fully restored

All lifecycle events are appended to lifecycle_log.json for inspection.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Union

import torch


class GaussianLifecycleManager:
    RESTORE_RATE     = 0.20    # confidence gain per frame when object is visible
    DECAY_RATE       = 0.10    # confidence loss per frame when object is absent
    STALE_THRESHOLD  = 0.20    # confidence below which fading begins
    FADE_STEPS       = 5       # frames to go from full opacity → invisible

    def __init__(self, gaussians, log_path: Optional[str] = None):
        """
        Parameters
        ----------
        gaussians : GaussianModel instance (held as a weak reference; the manager
                    reads/writes gaussians._opacity.data directly).
        log_path  : path to write lifecycle_log.json (optional).
        """
        self.gaussians          = gaussians
        self.log_path           = log_path
        self.entries: Dict[int, dict] = {}
        self._next_id           = 0
        self._event_log: list   = []

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, gaussian_indices, frame_idx: int,
                 object_id: Optional[int] = None) -> int:
        """Create a lifecycle entry for a newly confirmed Gaussian group.

        Parameters
        ----------
        gaussian_indices : torch.Tensor or list — Gaussian indices to track.
        frame_idx        : Frame at which the object was first confirmed.
        object_id        : Optional link to an ObjectMemory object_id.

        Returns
        -------
        entry_id (int) — key into self.entries.
        """
        entry_id = self._next_id
        self._next_id += 1

        idx_list = (gaussian_indices.cpu().tolist()
                    if hasattr(gaussian_indices, "cpu")
                    else list(gaussian_indices))

        # Store original logit-space opacities for accurate fade/restore
        with torch.no_grad():
            t_idx        = torch.tensor(idx_list, dtype=torch.long, device="cuda")
            orig_opacity = self.gaussians._opacity.data[t_idx].clone().cpu()

        self.entries[entry_id] = {
            "entry_id":          entry_id,
            "object_id":         object_id,
            "gaussian_indices":  idx_list,
            "age":               0,
            "visibility_count":  1,
            "last_observed_frame": frame_idx,
            "confidence":        1.0,
            "fading":            False,
            "fade_step":         0,
            # stored in logit space (inverse-sigmoid of actual opacity)
            "_orig_opacity_logit": orig_opacity.tolist(),
        }
        self._log("register", entry_id, frame_idx,
                  f"object_id={object_id}  n_gaussians={len(idx_list)}")
        return entry_id

    def update_seen(self, entry_id: int, frame_idx: int) -> None:
        """Mark the object as visible this frame and restore confidence / opacity."""
        if entry_id not in self.entries:
            return
        e = self.entries[entry_id]
        e["visibility_count"]   += 1
        e["last_observed_frame"]  = frame_idx
        e["confidence"]           = min(1.0, e["confidence"] + self.RESTORE_RATE)

        if e["fading"] and e["fade_step"] > 0:
            self._restore_opacity(entry_id)
            e["fading"]    = False
            e["fade_step"] = 0
            self._log("restored", entry_id, frame_idx,
                      f"re-observed, opacity restored  conf={e['confidence']:.2f}")
        else:
            self._log("update_seen", entry_id, frame_idx,
                      f"conf={e['confidence']:.2f}  vis={e['visibility_count']}")

    def tick(self, current_frame: int) -> None:
        """Advance one frame: age all entries, decay absent ones, fade stale ones.

        Call this once per frame *after* all update_seen() calls for that frame.
        """
        for entry_id, e in self.entries.items():
            e["age"] += 1
            if e["last_observed_frame"] >= current_frame:
                continue   # already handled via update_seen this frame

            # Object not seen this frame — decay
            e["confidence"] = max(0.0, e["confidence"] - self.DECAY_RATE)

            if e["confidence"] < self.STALE_THRESHOLD:
                if not e["fading"]:
                    e["fading"]    = True
                    e["fade_step"] = 0
                    self._log("stale", entry_id, current_frame,
                              f"conf={e['confidence']:.2f}, beginning fade")

                if e["fade_step"] < self.FADE_STEPS:
                    self._apply_fade_step(entry_id)
                    e["fade_step"] += 1
                    self._log("fade", entry_id, current_frame,
                              f"step={e['fade_step']}/{self.FADE_STEPS}  "
                              f"conf={e['confidence']:.2f}")
                else:
                    self._log("faded_out", entry_id, current_frame,
                              "fully faded — Gaussians invisible but not deleted")

        self._flush_log()

    def get_stale_indices(self) -> List[int]:
        """Return all Gaussian indices currently in a stale (fading) state."""
        result = []
        for e in self.entries.values():
            if e["confidence"] < self.STALE_THRESHOLD:
                result.extend(e["gaussian_indices"])
        return result

    def summary(self) -> str:
        lines = [f"GaussianLifecycleManager — {len(self.entries)} entries"]
        for eid, e in sorted(self.entries.items()):
            state = "FADING" if e["fading"] else ("stale" if e["confidence"] < self.STALE_THRESHOLD else "active")
            lines.append(
                f"  [{eid}] obj={e['object_id']}  "
                f"age={e['age']}  vis={e['visibility_count']}  "
                f"conf={e['confidence']:.2f}  "
                f"fade_step={e['fade_step']}/{self.FADE_STEPS}  "
                f"[{state}]"
            )
        return "\n".join(lines)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _apply_fade_step(self, entry_id: int) -> None:
        """Reduce opacity by one fade step (linear scale toward zero)."""
        e       = self.entries[entry_id]
        step    = e["fade_step"]                       # 0 → 4
        scale   = 1.0 - (step + 1) / self.FADE_STEPS  # 0.8, 0.6, 0.4, 0.2, 0.0

        t_idx        = torch.tensor(e["gaussian_indices"], dtype=torch.long, device="cuda")
        orig_logit   = torch.tensor(e["_orig_opacity_logit"], device="cuda")
        new_opacity  = torch.sigmoid(orig_logit) * scale
        new_opacity  = new_opacity.clamp(1e-4, 1 - 1e-4)

        with torch.no_grad():
            self.gaussians._opacity.data[t_idx] = torch.logit(new_opacity)

    def _restore_opacity(self, entry_id: int) -> None:
        """Restore Gaussians to their original opacity values."""
        e          = self.entries[entry_id]
        t_idx      = torch.tensor(e["gaussian_indices"], dtype=torch.long, device="cuda")
        orig_logit = torch.tensor(e["_orig_opacity_logit"], device="cuda")
        with torch.no_grad():
            self.gaussians._opacity.data[t_idx] = orig_logit

    def _log(self, event: str, entry_id: int, frame: int, detail: str = "") -> None:
        self._event_log.append({
            "frame": frame, "event": event,
            "entry_id": entry_id, "detail": detail,
        })

    def _flush_log(self) -> None:
        if self.log_path is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

        # Serialisable entries (exclude raw opacity tensors)
        serial_entries = {}
        for eid, e in self.entries.items():
            serial_entries[str(eid)] = {
                k: v for k, v in e.items() if k != "_orig_opacity_logit"
            }

        payload = {
            "entries":       serial_entries,
            "recent_events": self._event_log[-500:],   # cap log size
        }
        with open(self.log_path, "w") as f:
            json.dump(payload, f, indent=2)
