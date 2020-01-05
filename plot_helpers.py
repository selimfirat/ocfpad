# Reference: https://gitlab.idiap.ch/biometric-resources/lab-pad/blob/master/notebook/plot.py
import numpy
import bob.measure
from matplotlib import pyplot


def plot_scores_distributions(scores_dev, scores_eval, path, title='Score Distribution', n_bins=50, threshold_height=1,
                              legend_loc='best'):
    """
      Parameters
      ----------
      scores_dev : list
        The list containing negative and positive scores for the dev set
      scores_eval : list
        The list containing negative and positive scores for the eval set
      title: string
        Title of the plot
      n_bins: int
        Number of bins in the histogram
    """

    # compute the threshold on the dev set
    neg_dev = scores_dev[0]
    pos_dev = scores_dev[1]
    threshold = bob.measure.eer_threshold(scores_dev[0], scores_dev[1])

    f, ax = pyplot.subplots(1, 2, figsize=(15, 5))

    f.suptitle(title, fontsize=20)
    ax[0].hist(scores_dev[1], density=False, color='C1', bins=n_bins, label='Bona-fide')
    ax[0].hist(scores_dev[0], density=False, color='C7', bins=n_bins, alpha=0.4, hatch='\\\\',
               label='Presentation Attack')
    ax[0].vlines(threshold, 0, threshold_height, colors='r', linestyles='dashed', label='EER Threshold')
    ax[0].set_title('Development set')
    ax[0].set_xlabel("Score Value")
    ax[0].set_ylabel("Probability Density")
    ax[0].legend(loc=legend_loc)
    ax[1].hist(scores_eval[1], density=False, color='C1', bins=n_bins, label='Bona-fide')
    ax[1].hist(scores_eval[0], density=False, color='C7', bins=n_bins, alpha=0.4, hatch='\\\\',
               label='Presentation Attack')
    ax[1].vlines(threshold, 0, threshold_height, colors='r', linestyles='dashed', label='EER Threshold')
    ax[1].set_title('Evaluation set')
    ax[1].set_xlabel("Score Value")
    ax[1].set_ylabel("Probability Density")
    ax[1].legend(loc=legend_loc)

    pyplot.savefig(path)

    pyplot.clf()
