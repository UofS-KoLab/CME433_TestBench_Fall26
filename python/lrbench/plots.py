import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def strip(path, title, ids, columns):
    fig, ax = plt.subplots(len(ids), len(columns), figsize=(1.3 * len(columns), 1.35 * len(ids)))
    for r, i in enumerate(ids):
        for c, (label, imgs, score) in enumerate(columns):
            a = ax[r, c]
            a.imshow(imgs[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            a.set_xticks([])
            a.set_yticks([])
            if r == 0:
                a.set_title(label, fontsize=8)
            if score is not None:
                a.set_xlabel("%.1f dB" % score[i], fontsize=7, labelpad=1)
        ax[r, 0].set_ylabel("#%d" % i, fontsize=7)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=140)
    plt.close()
