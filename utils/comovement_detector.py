"""
Co-Movement Detection Module

Detects when a person (class 0) is moving together with other objects across multiple frames.
Uses spatial proximity analysis to identify and track person-object associations.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional


class CoMovementDetector:
    """
    Detects and tracks associations between persons and other objects based on spatial proximity.

    An association is confirmed when a person and object remain within proximity_threshold
    distance for at least min_frames consecutive frames.
    """

    def __init__(
        self,
        proximity_threshold: float = 150.0,
        min_frames: int = 10,
        max_gap_frames: int = 5,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the co-movement detector.

        Args:
            proximity_threshold: Maximum distance (pixels) between centers for association
            min_frames: Minimum consecutive frames to confirm association
            max_gap_frames: Maximum frames allowed with broken proximity before clearing association
            class_names: Dictionary mapping class IDs to names for logging
        """
        self.proximity_threshold = proximity_threshold
        self.min_frames = min_frames
        self.max_gap_frames = max_gap_frames
        self.class_names = class_names or {}

        # Store active associations: (person_track_id, object_track_id) -> association_data
        self.associations: Dict[Tuple[int, int], Dict] = {}

        # Store confirmed associations for current frame
        self.current_confirmed: Set[Tuple[int, int]] = set()

        # Track which associations have been logged to avoid duplicate logs
        self.logged_associations: Set[Tuple[int, int]] = set()

    def update(self, tracks: List[Tuple], frame_idx: int) -> List[Dict]:
        """
        Update associations based on current frame tracks.

        Args:
            tracks: List of tracks [(x1, y1, x2, y2, track_id, cls_id), ...]
            frame_idx: Current frame index

        Returns:
            List of confirmed associations with visualization data
        """
        # Separate persons (class 0) from other objects
        persons = [t for t in tracks if t[5] == 0]
        objects = [t for t in tracks if t[5] != 0]

        if not persons or not objects:
            # No potential associations, clean up old ones
            self._cleanup_stale_associations(frame_idx)
            return []

        # Find close proximity pairs
        current_pairs = self._find_proximity_pairs(persons, objects)

        # Update association tracking
        self._update_associations(current_pairs, frame_idx)

        # Clean up broken associations
        self._cleanup_stale_associations(frame_idx)

        # Build visualization data for confirmed associations
        viz_data = self._build_visualization_data(tracks)

        return viz_data

    def _find_proximity_pairs(
        self,
        persons: List[Tuple],
        objects: List[Tuple]
    ) -> Set[Tuple[int, int, int]]:
        """
        Find person-object pairs within proximity threshold.

        Args:
            persons: List of person tracks
            objects: List of object tracks

        Returns:
            Set of (person_track_id, object_track_id, object_class_id) tuples
        """
        pairs = set()

        for person in persons:
            p_x1, p_y1, p_x2, p_y2, p_id, _ = person
            p_center_x = (p_x1 + p_x2) / 2
            p_center_y = (p_y1 + p_y2) / 2

            for obj in objects:
                o_x1, o_y1, o_x2, o_y2, o_id, o_cls = obj
                o_center_x = (o_x1 + o_x2) / 2
                o_center_y = (o_y1 + o_y2) / 2

                # Calculate Euclidean distance between centers
                distance = np.sqrt(
                    (p_center_x - o_center_x) ** 2 +
                    (p_center_y - o_center_y) ** 2
                )

                if distance <= self.proximity_threshold:
                    pairs.add((p_id, o_id, o_cls))

        return pairs

    def _update_associations(
        self,
        current_pairs: Set[Tuple[int, int, int]],
        frame_idx: int
    ) -> None:
        """
        Update association history based on current frame pairs.

        Args:
            current_pairs: Set of (person_id, object_id, object_class) tuples in proximity
            frame_idx: Current frame index
        """
        # Update existing and new associations
        for person_id, object_id, object_class in current_pairs:
            pair_key = (person_id, object_id)

            if pair_key in self.associations:
                # Update existing association
                assoc = self.associations[pair_key]
                assoc['frames_together'] += 1
                assoc['last_frame'] = frame_idx
                assoc['gap_frames'] = 0  # Reset gap counter

                # Check if association is now confirmed
                if assoc['frames_together'] >= self.min_frames:
                    if not assoc['confirmed']:
                        assoc['confirmed'] = True
                        self._log_new_association(person_id, object_id, object_class, frame_idx)
                    self.current_confirmed.add(pair_key)
            else:
                # Create new association
                self.associations[pair_key] = {
                    'person_id': person_id,
                    'object_id': object_id,
                    'object_class': object_class,
                    'frames_together': 1,
                    'first_frame': frame_idx,
                    'last_frame': frame_idx,
                    'confirmed': False,
                    'gap_frames': 0
                }

        # Track pairs that broke proximity but haven't exceeded gap threshold
        current_pair_keys = {(p_id, o_id) for p_id, o_id, _ in current_pairs}

        for pair_key in list(self.associations.keys()):
            if pair_key not in current_pair_keys:
                # Pair is not in proximity this frame
                self.associations[pair_key]['gap_frames'] += 1

                # Remove from current confirmed if not in proximity
                self.current_confirmed.discard(pair_key)

    def _cleanup_stale_associations(self, frame_idx: int) -> None:
        """
        Remove associations that have exceeded the gap threshold.

        Args:
            frame_idx: Current frame index
        """
        to_remove = []

        for pair_key, assoc in self.associations.items():
            if assoc['gap_frames'] > self.max_gap_frames:
                to_remove.append(pair_key)
                # Remove from logged set to allow re-logging if they associate again
                self.logged_associations.discard(pair_key)

        for pair_key in to_remove:
            del self.associations[pair_key]
            self.current_confirmed.discard(pair_key)

    def _log_new_association(
        self,
        person_id: int,
        object_id: int,
        object_class: int,
        frame_idx: int
    ) -> None:
        """
        Log a newly confirmed association.

        Args:
            person_id: Person track ID
            object_id: Object track ID
            object_class: Object class ID
            frame_idx: Frame where association was confirmed
        """
        pair_key = (person_id, object_id)

        # Only log once per association
        if pair_key not in self.logged_associations:
            class_name = self.class_names.get(object_class, f"class_{object_class}")
            print(f"[Frame {frame_idx}] Person identified with {class_name} (Person ID:{person_id}, Object ID:{object_id})")
            self.logged_associations.add(pair_key)

    def _build_visualization_data(self, tracks: List[Tuple]) -> List[Dict]:
        """
        Build visualization data for confirmed associations.

        Args:
            tracks: All current tracks

        Returns:
            List of dictionaries with visualization information
        """
        viz_data = []

        # Create track lookup dictionary
        track_dict = {track[4]: track for track in tracks}

        for pair_key in self.current_confirmed:
            person_id, object_id = pair_key

            # Get track data
            if person_id not in track_dict or object_id not in track_dict:
                continue

            person_track = track_dict[person_id]
            object_track = track_dict[object_id]

            assoc = self.associations[pair_key]

            viz_data.append({
                'person_id': person_id,
                'object_id': object_id,
                'person_bbox': person_track[:4],
                'object_bbox': object_track[:4],
                'object_class': assoc['object_class'],
                'frames_together': assoc['frames_together']
            })

        return viz_data

    def get_statistics(self) -> Dict:
        """
        Get statistics about current associations.

        Returns:
            Dictionary with statistics
        """
        total_associations = len(self.associations)
        confirmed_associations = len(self.current_confirmed)
        pending_associations = total_associations - confirmed_associations

        return {
            'total_associations': total_associations,
            'confirmed_associations': confirmed_associations,
            'pending_associations': pending_associations,
            'logged_associations': len(self.logged_associations)
        }
