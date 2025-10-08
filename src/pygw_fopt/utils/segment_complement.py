import json
import matplotlib.pyplot as plt

class SegmentComplement:
    """
    A class to manage and compute complementary segments for one or more main segments.
    Parameters:
        bound_segments (tuple or list of tuples): 
            A single tuple (start, end) or a list of tuples representing the main segments.
    """
    def __init__(self, bound_segments):
        if isinstance(bound_segments, tuple):
            self.bound_segments = [bound_segments]
        else:
            self.bound_segments = bound_segments
        self.sub_segments_with_flags = []  # New attribute to store sub-segments with flag handling

    def update_bound_segments(self, new_bound_segments, reset_sub_segments=True):
        """
        Dynamically update the main segments.
        Parameters:
            new_bound_segments (tuple or list of tuples): 
                A single tuple (start, end) or a list of tuples representing the new main segments.
            reset_sub_segments (bool): If True, clears existing sub-segments.
        """
        if isinstance(new_bound_segments, tuple):
            self.bound_segments = [new_bound_segments]
        else:
            self.bound_segments = new_bound_segments
        if reset_sub_segments:
            self.sub_segments_with_flags = []

    def add_sub_segments(self, sub_segments, complement_flags=None, allow_out_of_bounds=False):
        """
        Add sub-segments to the collection, handling unsorted input and optionally allowing out-of-bounds sub-segments.
        Parameters:
            sub_segments (list of tuples): A list of tuples [(start1, end1), (start2, end2), ...].
            complement_flags (list of bool or None): A list of boolean flags corresponding to each sub-segment.
                                                    True indicates the segment is added directly.
                                                    False indicates the complement of the segment is added.
                                                    If None, all flags are set to True by default.
            allow_out_of_bounds (bool): If True, allows sub-segments to extend beyond the main segment bounds.
                                        If False, raises an error for sub-segments that violate the bounds.
        """
        def normalize_sub_segment(sub):
            """Ensure start <= end for each sub-segment."""
            sub_start, sub_end = sub
            return (min(sub_start, sub_end), max(sub_start, sub_end))

        normalized_sub_segments = [normalize_sub_segment(sub) for sub in sub_segments]

        # Assign default complement_flags if None is provided
        if complement_flags is None:
            complement_flags = [True] * len(normalized_sub_segments)

        # Track original sub-segments with True flags
        self._original_true_sub_segments = []

        for sub, flag in zip(normalized_sub_segments, complement_flags):
            if flag:  # If flag is True, append the sub-segment directly
                self.sub_segments_with_flags.append(sub)
                self._original_true_sub_segments.append(sub)  # Track original True sub-segments
            else:  # If flag is False, compute the complement and append it
                complement = self._compute_complement_for_single_segment(sub)
                self.sub_segments_with_flags.extend(complement)

        # Sort the updated sub_segments_with_flags
        self.sub_segments_with_flags.sort(key=lambda x: x[0])

    # def add_sub_segments(self, sub_segments, complement_flags=None, allow_out_of_bounds=False):
    #     """
    #     Add sub-segments to the collection, handling unsorted input and optionally allowing out-of-bounds sub-segments.
    #     Parameters:
    #         sub_segments (list of tuples): A list of tuples [(start1, end1), (start2, end2), ...].
    #         complement_flags (list of bool or None): A list of boolean flags corresponding to each sub-segment.
    #                                                  True indicates the segment is added directly.
    #                                                  False indicates the complement of the segment is added.
    #                                                  If None, all flags are set to True by default.
    #         allow_out_of_bounds (bool): If True, allows sub-segments to extend beyond the main segment bounds.
    #                                     If False, raises an error for sub-segments that violate the bounds.
    #     """
    #     def normalize_sub_segment(sub):
    #         """Ensure start <= end for each sub-segment."""
    #         sub_start, sub_end = sub
    #         return (min(sub_start, sub_end), max(sub_start, sub_end))

    #     normalized_sub_segments = [normalize_sub_segment(sub) for sub in sub_segments]

    #     # Assign default complement_flags if None is provided
    #     if complement_flags is None:
    #         complement_flags = [True] * len(normalized_sub_segments)

    #     if not allow_out_of_bounds:
    #         invalid_sub_segments = []
    #         for sub, flag in zip(normalized_sub_segments, complement_flags):
    #             if not any(sub[0] >= bound[0] and sub[1] <= bound[1] for bound in self.bound_segments):
    #                 invalid_sub_segments.append((sub, flag))
    #         if invalid_sub_segments:
    #             invalid_details = ", ".join([f"{sub} (flag={flag})" for sub, flag in invalid_sub_segments])
    #             raise ValueError(
    #                 f"The following sub-segments are out of bounds for the main segments {self.bound_segments}: {invalid_details}. "
    #                 "Set allow_out_of_bounds=True to allow these."
    #             )

    #     for sub, flag in zip(normalized_sub_segments, complement_flags):
    #         if flag:  # If flag is True, append the sub-segment directly
    #             self.sub_segments_with_flags.append(sub)
    #         else:  # If flag is False, compute the complement and append it
    #             complement = self._compute_complement_for_single_segment(sub)
    #             self.sub_segments_with_flags.extend(complement)

    #     # Sort the updated sub_segments_with_flags
    #     self.sub_segments_with_flags.sort(key=lambda x: x[0])

    def _compute_complement_for_single_segment(self, sub_segment):
        """
        Compute the complement of a single sub-segment with respect to the main segments.
        Parameters:
            sub_segment (tuple): A single sub-segment (start, end).
        Returns:
            A list of tuples representing the complementary segments.
        """
        complements = []
        for bound in self.bound_segments:
            current_start = bound[0]
            current_end = bound[1]
            if sub_segment[0] > current_start:
                complements.append((current_start, min(sub_segment[0], current_end)))
            if sub_segment[1] < current_end:
                complements.append((max(sub_segment[1], current_start), current_end))
        return complements

    def _generate_complementary_segments(self):
        """
        Internal generator to compute the complementary segments for all main segments.
        Yields:
            Tuples representing the complementary segments.
        """
        for bound in self.bound_segments:
            current_start = bound[0]
            relevant_sub_segments = [
                sub for sub in self.sub_segments_with_flags
                if sub[0] < bound[1] and sub[1] > bound[0]
            ]
            relevant_sub_segments.sort(key=lambda x: x[0])
            for sub in relevant_sub_segments:
                sub_start, sub_end = sub
                if current_start < sub_start:
                    yield (current_start, min(sub_start, bound[1]))
                current_start = max(current_start, sub_end)
            if current_start < bound[1]:
                yield (current_start, bound[1])

    def get_complementary_segments(self):
        """
        Compute the complementary segments of the main segments with respect to the sub-segments.
        Returns:
            A list of tuples representing the complementary segments.
        """
        return list(self._generate_complementary_segments())

    def merge_overlapping_sub_segments(self):
        """
        Merge overlapping sub-segments into a single continuous segment.
        """
        if not self.sub_segments_with_flags:
            return
        merged = []
        current_start, current_end = self.sub_segments_with_flags[0]
        for sub in self.sub_segments_with_flags[1:]:
            if sub[0] <= current_end:  # Overlap detected
                current_end = max(current_end, sub[1])  # Extend the current segment
            else:
                merged.append((current_start, current_end))
                current_start, current_end = sub
        merged.append((current_start, current_end))
        self.sub_segments_with_flags = merged

    def to_json(self):
        """
        Serialize the object to JSON.
        Returns:
            A JSON string representing the object.
        """
        return json.dumps({
            "bound_segments": self.bound_segments,
            "sub_segments_with_flags": self.sub_segments_with_flags
        })

    @classmethod
    def from_json(cls, json_str):
        """
        Deserialize the object from JSON.
        Parameters:
            json_str (str): A JSON string representing the object.
        Returns:
            A new SegmentComplement object.
        """
        data = json.loads(json_str)
        obj = cls(data["bound_segments"])
        obj.sub_segments_with_flags = data["sub_segments_with_flags"]
        return obj

    def visualize(self):
        """
        Visualize the main segments, sub-segments (with True/False flags on different heights), and complementary segments.
        Returns:
            fig, ax for further customization or saving.
        """
        fig, ax = plt.subplots(figsize=(10, 2))

        # Plot the main segments
        for i, bound in enumerate(self.bound_segments):
            ax.plot([bound[0], bound[1]], [1, 1], 'b-', linewidth=4, label='Main Segments' if i == 0 else "")

        # Separate sub-segments by their complement_flags
        true_sub_segments = [sub for sub in self._original_true_sub_segments]
        false_sub_segments = [
            sub for sub in self.sub_segments_with_flags 
            if sub not in self._original_true_sub_segments
        ]

        # Plot sub-segments with complement_flags=True (e.g., red at y=0.7)
        for i, sub in enumerate(true_sub_segments):
            ax.plot([sub[0], sub[1]], [0.7, 0.7], 'r-', linewidth=3, label='Sub-Segments (True)' if i == 0 else "")

        # Plot sub-segments with complement_flags=False (e.g., yellow at y=0.3)
        for i, sub in enumerate(false_sub_segments):
            ax.plot([sub[0], sub[1]], [0.3, 0.3], 'y-', linewidth=3, label='Sub-Segments (False)' if i == 0 else "")

        # Plot the complementary segments
        for i, comp in enumerate(self.get_complementary_segments()):
            ax.plot([comp[0], comp[1]], [0, 0], 'g-', linewidth=3, label='Complementary Segments' if i == 0 else "")

        # Add legend and formatting
        ax.set_yticks([])
        ax.set_title("Segment Visualization")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xlim(min(b[0] for b in self.bound_segments) - 1, max(b[1] for b in self.bound_segments) + 1)
        ax.grid(True)

        return fig, ax