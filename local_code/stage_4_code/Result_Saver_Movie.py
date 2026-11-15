"""Save portable JSON/CSV results through the provided ResultModule interface."""
import csv
import json
from pathlib import Path
from local_code.base_class.result import result


class Result_Saver_Movie(result):
    def __init__(self, folder):
        super().__init__('movie results', 'JSON metrics and CSV history/predictions')
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def save(self):
        (self.folder/'metrics.json').write_text(json.dumps(self.data['metrics'], indent=2)+'\n')
        for name in ('history','predictions'):
            rows=self.data[name]
            if rows:
                with (self.folder/f'{name}.csv').open('w', newline='') as f:
                    writer=csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)

    def load(self):
        return json.loads((self.folder/'metrics.json').read_text())
