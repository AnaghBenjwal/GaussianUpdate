"""
object_memory.py — Persistent Object Memory Module

Maintains a per-object registry across frames. Each entry tracks Gaussian
membership, detection history, and a confidence score that gates whether a
detected displacement is committed to the 3D model.

Confidence logic:
  - New object detected:  confidence = 0.3  (provisional)
  - Seen again (match):   confidence += 0.2, capped at 1.0
  - Commit gate:          confidence > 0.7  (seen at least ~3 times consistently)

Matching is done by IoU of Gaussian index sets: if the detected Gaussian group
overlaps the stored group by >= MATCH_IOU_THRESHOLD, it's treated as the same
object. Falls back to no-match (new entry) if no existing entry qualifies.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


class ObjectMemory:
    CONFIDENCE_INIT       = 0.3
    CONFIDENCE_INCREMENT  = 0.2
    CONFIDENCE_MAX        = 1.0
    COMMIT_THRESHOLD      = 0.7
    MATCH_IOU_THRESHOLD   = 0.10   # Gaussian-index IoU to consider a match

    def __init__(self, save_path: Optional[str] = None):
        """
        Parameters
        ----------
        save_path : path to write memory.json after each update (optional).
        """
        self.entries:   dict[int, dict] = {}
        self._next_id:  int             = 0
        self.save_path: Optional[str]   = save_path

    # ── Public API ────────────────────────────────────────────────────────────

    def associate(
        self,
        gaussian_indices,
        semantic_label:  str,
        frame_idx:       int,
        position_xyz:    Optional[Union[list, tuple]] = None,
    ) -> Tuple[int, bool]:
        """Match a detected Gaussian group to an existing memory entry or create one.

        Parameters
        ----------
        gaussian_indices : torch.Tensor or list of int — Gaussian indices for the detection.
        semantic_label   : Human-readable class name (e.g. "apple", "bowl").
        frame_idx        : Current frame / call index.
        position_xyz     : Mean 3-D world position of the object (optional; appended to
                           position_history if provided).

        Returns
        -------
        (object_id, is_new) where is_new=True if a fresh entry was created.
        """
        idx_set = self._to_set(gaussian_indices)
        best_id, best_iou = self._find_best_match(idx_set)

        if best_id is not None and best_iou >= self.MATCH_IOU_THRESHOLD:
            self._update_existing(best_id, idx_set, frame_idx, position_xyz)
            if self.save_path:
                self.save()
            return best_id, False

        # New entry
        new_id = self._next_id
        self._next_id += 1
        pos_hist = [list(position_xyz)] if position_xyz is not None else []
        self.entries[new_id] = {
            "object_id":        new_id,
            "semantic_label":   semantic_label,
            "gaussian_indices": list(idx_set),
            "confidence":       self.CONFIDENCE_INIT,
            "last_seen_frame":  frame_idx,
            "position_history": pos_hist,
            "detection_count":  1,
        }
        if self.save_path:
            self.save()
        return new_id, True

    def should_commit(self, object_id: int) -> bool:
        """Return True if the object has enough evidence to apply a 3D update."""
        entry = self.entries.get(object_id)
        if entry is None:
            return False
        return entry["confidence"] >= self.COMMIT_THRESHOLD

    def get_entry(self, object_id: int) -> Optional[dict]:
        return self.entries.get(object_id)

    def save(self, path: Optional[str] = None) -> None:
        """Write current memory state to JSON."""
        target = path or self.save_path
        if target is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "w") as f:
            json.dump(
                {"_next_id": self._next_id, "entries": self.entries},
                f, indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "ObjectMemory":
        """Restore a previously saved memory state."""
        m = cls(save_path=path)
        with open(path) as f:
            data = json.load(f)
        m.entries   = {int(k): v for k, v in data["entries"].items()}
        m._next_id  = data.get("_next_id", len(m.entries))
        return m

    def summary(self) -> str:
        lines = [f"ObjectMemory — {len(self.entries)} entries"]
        for eid, e in sorted(self.entries.items()):
            commit = "COMMIT-OK" if e["confidence"] >= self.COMMIT_THRESHOLD else "below-threshold"
            lines.append(
                f"  [{eid}] {e['semantic_label']:<16}  "
                f"conf={e['confidence']:.2f}  "
                f"seen={e['detection_count']}x  "
                f"last_frame={e['last_seen_frame']}  "
                f"({commit})"
            )
        return "\n".join(lines)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _find_best_match(self, idx_set: set) -> Tuple[Optional[int], float]:
        best_id, best_iou = None, 0.0
        for eid, entry in self.entries.items():
            iou = self._gaussian_iou(idx_set, set(entry["gaussian_indices"]))
            if iou > best_iou:
                best_iou = iou
                best_id  = eid
        return best_id, best_iou

    def _update_existing(
        self,
        object_id:   int,
        idx_set:     set,
        frame_idx:   int,
        position_xyz,
    ) -> None:
        e = self.entries[object_id]
        e["confidence"]       = min(self.CONFIDENCE_MAX,
                                    e["confidence"] + self.CONFIDENCE_INCREMENT)
        e["detection_count"] += 1
        e["last_seen_frame"]  = frame_idx
        e["gaussian_indices"] = list(idx_set)
        if position_xyz is not None:
            e["position_history"].append(list(position_xyz))

    @staticmethod
    def _gaussian_iou(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _to_set(gaussian_indices) -> set:
        if hasattr(gaussian_indices, "tolist"):
            return set(gaussian_indices.tolist())
        return set(gaussian_indices)
