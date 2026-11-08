## Configuration experiments (compact, for the report)

| Configuration | ORL test acc % | CIFAR test acc % |
|---|---|---|
| Baseline | 96.7 ± 1.4 | 76.0 |
| Depth: one fewer conv block | 95.8 ± 1.4 (-0.8) | 73.5 (-2.5) |
| Depth: one more conv block | 95.0 ± 4.3 (-1.7) | 75.1 (-0.9) |
| Kernel size 5x5 (padding 2) | 97.5 ± 2.5 (+0.8) | 76.1 (+0.1) |
| Padding 0 | 93.3 ± 1.4 (-3.3) | 72.7 (-3.3) |
| Stride 2 conv instead of pooling | 93.3 ± 1.4 (-3.3) | 69.6 (-6.4) |
| Average pooling | 95.8 ± 1.4 (-0.8) | 77.0 (+0.9) |
| Hidden layer 64 units | 95.0 ± 2.5 (-1.7) | 74.3 (-1.8) |
| Add batch normalization | 97.5 ± 0.0 (+0.8) | 78.9 (+2.9) |
| No dropout | 99.2 ± 1.4 (+2.5) | 73.0 (-3.1) |
| Loss: multi-class margin | 97.5 ± 0.0 (+0.8) | 75.3 (-0.7) |
| Optimizer: SGD (lr 0.01, momentum 0.9) | 95.8 ± 3.8 (-0.8) | 76.9 (+0.8) |

Test accuracy in %, difference from the baseline in parentheses; "± x" is the standard deviation over seeds. Configurations were compared on the validation split (full table below); the supplied test split was evaluated once per run.

## Full table (test and validation accuracy)

| Configuration | ORL test acc % | ORL val acc % | CIFAR test acc % | CIFAR val acc % |
|---|---|---|---|---|
| Baseline | 96.7 ± 1.4 | 98.3 ± 1.4 | 76.0 | 76.4 |
| Depth: one fewer conv block | 95.8 ± 1.4 (-0.8) | 95.8 ± 1.4 | 73.5 (-2.5) | 73.3 |
| Depth: one more conv block | 95.0 ± 4.3 (-1.7) | 97.5 ± 2.5 | 75.1 (-0.9) | 76.3 |
| Kernel size 5x5 (padding 2) | 97.5 ± 2.5 (+0.8) | 95.8 ± 2.9 | 76.1 (+0.1) | 75.7 |
| Padding 0 | 93.3 ± 1.4 (-3.3) | 95.8 ± 2.9 | 72.7 (-3.3) | 73.8 |
| Stride 2 conv instead of pooling | 93.3 ± 1.4 (-3.3) | 95.0 ± 2.5 | 69.6 (-6.4) | 69.9 |
| Average pooling | 95.8 ± 1.4 (-0.8) | 97.5 ± 2.5 | 77.0 (+0.9) | 77.2 |
| Hidden layer 64 units | 95.0 ± 2.5 (-1.7) | 97.5 ± 2.5 | 74.3 (-1.8) | 74.5 |
| Add batch normalization | 97.5 ± 0.0 (+0.8) | 99.2 ± 1.4 | 78.9 (+2.9) | 78.9 |
| No dropout | 99.2 ± 1.4 (+2.5) | 100.0 ± 0.0 | 73.0 (-3.1) | 73.2 |
| Loss: multi-class margin | 97.5 ± 0.0 (+0.8) | 98.3 ± 1.4 | 75.3 (-0.7) | 75.7 |
| Optimizer: SGD (lr 0.01, momentum 0.9) | 95.8 ± 3.8 (-0.8) | 99.2 ± 1.4 | 76.9 (+0.8) | 75.8 |
