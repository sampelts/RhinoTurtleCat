'''
Base IO class for history
'''

import abc

class history:
    '''
        HistoryModule: to keep track of accuracy and loss during training
        entries: history name
                history description
    '''
    data = None

    history_name = None
    history_description = None

    history_destination_folder_path = None
    history_destination_file_name = None

    def __init__(self, hName=None, hDescription=None):
        self.data = []
        self.history_name = hName
        self.history_description = hDescription

    def record(self, **kwargs):
        self.data.append(kwargs)

    @abc.abstractmethod
    def save(self):
        return

    @abc.abstractmethod
    def load(self):
        return
