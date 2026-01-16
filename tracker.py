# -*- coding: utf-8 -*-
"""
tracker.py

Centroid-based multi-object tracker.

Responsibilities:
- Assign persistent IDs (tracklets)
- Maintain centroid history per ID
- Maintain per-frame centroid displacement history (for binary motion detection)
- Handle temporary disappearance (MAX_LOST_FRAMES)

This module intentionally does NOT:
- compute velocity or speed
- accumulate morphology statistics
- infer behavior or polarity

Those are handled downstream.
"""

import numpy as np
import config


class CentroidTracker:
    def __init__(self, max_lost=None):
        self.next_object_id = 0
        self.objects = {}   # object_id -> track dict
        self.lost = {}      # object_id -> lost frame count

        self.max_lost = (
            max_lost if max_lost is not None else config.MAX_LOST_FRAMES
        )

    @staticmethod
    def _euclid(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def update(self, detections):
        """
        Assign IDs to detections and update track histories.

        Parameters
        ----------
        detections : list of dict
            Each dict must contain:
                - "centroid": (x, y)

        Returns
        -------
        updated_detections : list of dict
            Same dicts with "id" field added.
        """

        updated = []
        used_ids = set()

        # --------------------------------------------------
        # Match detections to existing tracks
        # --------------------------------------------------
        for det in detections:
            c = det["centroid"]

            best_id = None
            best_dist = config.MAX_TRACK_DIST

            for oid, obj in self.objects.items():
                if oid in used_ids:
                    continue

                d = self._euclid(c, obj["centroid"])
                if d < best_dist:
                    best_dist = d
                    best_id = oid

            if best_id is not None:
                # Update existing track
                track = self.objects[best_id]

                # Per-frame centroid displacement
                prev = track["centroid"]
                delta = self._euclid(c, prev)

                track["recent_deltas"].append(delta)
                track["recent_deltas"] = track["recent_deltas"][
                    -config.DETECT_WINDOW_FRAMES:
                ]

                track["centroid"] = c
                track["history"].append(c)
                track["history"] = track["history"][-config.MAX_HISTORY:]

                self.lost[best_id] = 0
                det["id"] = best_id
                used_ids.add(best_id)

            else:
                # Create new track
                oid = self.next_object_id
                self.objects[oid] = {
                    "centroid": c,
                    "history": [c],
                    "recent_deltas": [],
                }
                self.lost[oid] = 0
                det["id"] = oid
                used_ids.add(oid)

                self.next_object_id += 1

            updated.append(det)

        # --------------------------------------------------
        # Handle lost tracks
        # --------------------------------------------------
        current_ids = {d["id"] for d in updated}

        for oid in list(self.objects.keys()):
            if oid not in current_ids:
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    del self.objects[oid]
                    del self.lost[oid]

        return updated
