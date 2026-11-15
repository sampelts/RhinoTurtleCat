"""Draw the exact shared movie classifier architecture for the report."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.movie_plot_results import plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 2.1))
    boxes = [
        (0.15, 1.65, 'Review tokens', 'B × T indices\nT ≤ 300'),
        (2.2, 1.8, 'Embedding', '25,000 × 128\ntrainable vectors'),
        (4.4, 2.4, 'LSTM or GRU', '128 hidden units\nlast real token → B × 128'),
        (7.2, 2.4, 'Binary classifier', 'Linear 128 → 1\nsigmoid → positive probability'),
    ]
    for x, width, title, detail in boxes:
        box = FancyBboxPatch((x, .45), width, 1.05, boxstyle='round,pad=0.06',
                             facecolor='#eef3f7', edgecolor='#35546c', linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + width/2, 1.25, title, ha='center', va='center', fontsize=11, weight='bold')
        ax.text(x + width/2, .82, detail, ha='center', va='center', fontsize=9)
    for previous, following in zip(boxes, boxes[1:]):
        x, width, _, _ = previous
        next_x = following[0]
        ax.add_patch(FancyArrowPatch((x + width + .08, .98), (next_x - .1, .98),
                                    arrowstyle='-|>', mutation_scale=12, color='#35546c'))
    ax.text(4.95, .12, 'Positive if probability ≥ 0.5; right padding is excluded from the review representation.',
            ha='center', fontsize=9, color='#444444')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.8)
    ax.axis('off')
    fig.tight_layout(pad=.3)
    fig.savefig(output, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    draw(ROOT / 'report/movie_lstm_gru/architecture.png')
