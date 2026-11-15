"""LSTM counterpart of Madison's Movie_RNN, with the same default dimensions."""
from local_code.stage_4_code.Method_Recurrent_Movie import MovieRecurrent


class Movie_LSTM(MovieRecurrent):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, cell='LSTM', **kwargs)
