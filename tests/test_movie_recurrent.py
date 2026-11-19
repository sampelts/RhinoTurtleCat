"""Verify padding handling and gradient propagation for LSTM and GRU."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from local_code.stage_4_code.Method_LSTM_Movie import Movie_LSTM
from local_code.stage_4_code.Method_GRU_Movie import Movie_GRU


class RecurrentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        torch.set_num_threads(2)
        self.x = torch.tensor([[2, 3, 4, 5, 0], [6, 7, 0, 0, 0], [8, 9, 10, 0, 0]])
        self.lengths = torch.tensor([4, 2, 3])

    def test_true_length_outputs_match_packed_hidden_state(self):
        for cls in (Movie_LSTM, Movie_GRU):
            with self.subTest(model=cls.__name__):
                model = cls(20, embedding_dim=8, hidden_size=12).eval()
                with torch.no_grad():
                    actual = model(self.x, self.lengths)
                    packed = pack_padded_sequence(model.embedding(self.x), self.lengths,
                                                 batch_first=True, enforce_sorted=False)
                    _, state = model.recurrent(packed)
                    hidden = state[0] if isinstance(state, tuple) else state
                    expected = model.fc(hidden[-1]).squeeze(1)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_padding_and_batch_context_do_not_change_predictions(self):
        for cls in (Movie_LSTM, Movie_GRU):
            with self.subTest(model=cls.__name__):
                model = cls(20, embedding_dim=8, hidden_size=12).eval()
                with torch.no_grad():
                    original = model(self.x, self.lengths)
                    padded = model(torch.nn.functional.pad(self.x, (0, 7)), self.lengths)
                    alone = model(self.x[1:2, :2], self.lengths[1:2])
                torch.testing.assert_close(original, padded, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(original[1:2], alone, atol=1e-6, rtol=1e-5)

    def test_backward_reaches_gates_and_preserves_padding_embedding(self):
        for cls in (Movie_LSTM, Movie_GRU):
            with self.subTest(model=cls.__name__):
                model = cls(20, embedding_dim=8, hidden_size=12)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(self.x, self.lengths), torch.tensor([1., 0., 1.]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(float(model.recurrent.weight_ih_l0.grad.abs().sum()), 0.)
                self.assertEqual(float(model.embedding.weight.grad[0].abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
