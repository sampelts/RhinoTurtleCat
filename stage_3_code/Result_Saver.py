
"""Save experiment results using pickle."""

import pickle
from pathlib import Path

from local_code.base_class.result import result


class Result_Saver(result):

    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print("saving results...")

        # Get destination directory
        folder = Path(
            self.result_destination_folder_path or "."
        )

        # Create directory if it does not exist
        folder.mkdir(parents=True, exist_ok=True)

        # Construct result filename
        filename = (
            self.result_destination_file_name
            + "_"
            + str(self.fold_count)
        )

        full_path = folder / filename

        # Save experiment results
        with full_path.open("wb") as f:
            pickle.dump(self.data, f)

        print("results saved to:", full_path)
