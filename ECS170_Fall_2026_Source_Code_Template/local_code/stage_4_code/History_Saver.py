import csv
import os
from local_code.base_class.history import history

class History_Saver(history):

    def save(self):
        print('saving training history...')
        if not self.data:
            print('No history data to save.')
            return

        folder = self.history_destination_folder_path or ''
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        full_path = os.path.join(folder, self.history_destination_file_name)

        fieldnames = list(self.data[0].keys())
        with open(full_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)
        print('history saved to:', full_path)

    def load(self):
        folder = self.history_destination_folder_path or ''
        full_path = os.path.join(folder, self.history_destination_file_name)
        with open(full_path, 'w') as f:
            reader = csv.DictReader(f)
            self.data = [row for row in reader]
        return self.data