import numpy as np
from scipy.special import logit, expit
from scipy.optimize import minimize

import numpy as np
from scipy.special import logit
import itertools
import sklearn.linear_model as lm

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

np.random.seed(0)


class CVTMLE:
    def __init__(self, q_t0=None, q_t1=None, g=None, t=None, y=None, fromFolds=None, est_keys=None,
                 truncate_level=0.05):
        """
        CVTMLE as conceived by Levi, 2018:
        Levy, Jonathan. "An easy implementation of CV-TMLE." arXiv preprint arXiv:1811.04573 (2018).

        :param q_t0: initial estimate with control exposure
        :param q_t1: initial estimate with treatment/non-control exposure
        :param g: prediction of propensity score
        :param t: treatment label
        :param y: factual outcome
        :param fromFolds: if files for estimates per fold are provided (type list) which are npz files, then no need to provide first five parameterse
        :param est_keys: once npz files are read, the keys are needed to extract estimates (i.e., first five parameters in this init)
        :param truncate_level: truncation for propensity scores (0.05 default means that only patients with estimates between 0.05 and 0.95 will be considered)
        """


        self.q_t0 = q_t0
        self.q_t1 = q_t1
        self.g = g
        self.t = t
        self.y = y
        self.est_keys = est_keys
        self.truncate_level = truncate_level
        if fromFolds is not None:
            self.q_t0, self.q_t1, self.y, self.g, self.t = self.collateFromFolds(fromFolds)

    def _perturbed_model_bin_outcome(self, q_t0, q_t1, g, t, eps):
        """
        Helper for psi_tmle_bin_outcome

        Returns q_\eps (t,x) and the h term
        (i.e., value of perturbed predictor at t, eps, x; where q_t0, q_t1, g are all evaluated at x
        """
        h = t * (1. / g) - (1. - t) / (1. - g)
        full_lq = (1. - t) * logit(q_t0) + t * logit(q_t1)  # logit predictions from unperturbed model
        logit_perturb = full_lq + eps * h
        return expit(logit_perturb), h

    def run_tmle_binary(self):
        """
        This is for CV-TMLE on binary outcomes yielding risk ratio with 95% CI. Read Levi et al for methodological details.
        Influence curves coded from Gruber S, van der Laan, MJ. (2011).

        """

        print('running CV-TMLE for binary outcomes...')
        q_t0, q_t1, g, t, y, truncatel = np.copy(self.q_t0), np.copy(self.q_t1), np.copy(self.g), np.copy(
            self.t), np.copy(self.y), np.copy(self.truncate_level)
        q_t0, q_t1, g, t, y = self.truncate_all_by_g(q_t0, q_t1, g, t, y, truncatel)

        eps_hat = minimize(
            lambda eps: self.cross_entropy(y, self._perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps)[0]), 0.,
            method='Nelder-Mead')
        eps_hat = eps_hat.x[0]

        def q1(t_cf):
            return self._perturbed_model_bin_outcome(q_t0, q_t1, g, t_cf, eps_hat)

        qall = ((1. - t) * (q_t0)) + (t * (q_t1))  # full predictions from unperturbed model

        qq1, h1 = q1(np.ones_like(t))
        qq0, h0 = q1(np.zeros_like(t))
        rr = np.mean(qq1) / np.mean(qq0)

        ic = (1 / np.mean(qq1) * (h1 * (y - qall) + qq1 - np.mean(qq1)) -
              (1 / np.mean(qq0)) * (-1 * h0 * (y - qall) + qq0 - np.mean(qq0)))
        psi_tmle_std = 1.96 * np.sqrt(np.var(ic) / (t.shape[0]))

        return [rr, np.exp(np.log(rr) - psi_tmle_std), np.exp(np.log(rr) + psi_tmle_std)]

    def run_tmle_continuous(self):
        """
        This is for CV-TMLE on continuous outcomes yielding ATE/MD with 95% CI. Read Levi et al for methodological details.
        Influence curves coded from Gruber S, van der Laan, MJ. (2011).

        """
        print('running CV-TMLE for continuous outcomes...')

        q_t0, q_t1, g, t, y, truncatel = np.copy(self.q_t0), np.copy(self.q_t1), np.copy(self.g), np.copy(
            self.t), np.copy(self.y), np.copy(self.truncate_level)
        q_t0, q_t1, g, t, y = self.truncate_all_by_g(q_t0, q_t1, g, t, y, truncatel)

        h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
        full_q = (1.0 - t) * q_t0 + t * q_t1
        eps_hat = np.sum(h * (y - full_q)) / np.sum(np.square(h))

        def q1(t_cf):
            h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
            full_q = ((1.0 - t_cf) * q_t0) + (t_cf * q_t1)
            return full_q + eps_hat * h_cf, h_cf

        qq1, h_cf1 = q1(np.ones_like(t))
        qq0, h_cf0 = q1(np.zeros_like(t))
        haw = h_cf0 + h_cf1

        rd = np.mean(qq1 - qq0)
        ic = (haw * (y - full_q)) + (qq1 - qq0) - rd
        psi_tmle_std = 1.96 * np.sqrt(np.var(ic) / (t.shape[0]))

        return [rd, rd - psi_tmle_std, rd + psi_tmle_std]

    def truncate_by_g(self, attribute, g, level=0.1):
        keep_these = np.logical_and(g >= level, g <= 1. - level)
        return attribute[keep_these]

    def truncate_all_by_g(self, q_t0, q_t1, g, t, y, truncate_level=0.05):
        """
        Helper function to clean up nuisance parameter estimates.
        """
        orig_g = np.copy(g)
        q_t0 = self.truncate_by_g(np.copy(q_t0), orig_g, truncate_level)
        q_t1 = self.truncate_by_g(np.copy(q_t1), orig_g, truncate_level)
        g = self.truncate_by_g(np.copy(g), orig_g, truncate_level)
        t = self.truncate_by_g(np.copy(t), orig_g, truncate_level)
        y = self.truncate_by_g(np.copy(y), orig_g, truncate_level)
        return q_t0, q_t1, g, t, y

    def cross_entropy(self, y, p):
        return -np.mean((y * np.log(p) + (1. - y) * np.log(1. - p)))

    def collateFromFolds(self, foldNPZ):
        """
        FYI: keys can be provided but default is below
        est_keys = {
        'treatment_label_key' : 'treatment_label',
        'outcome_key' : 'outcome',
        'treatment_pred_key' : 'treatment',
        'outcome_label_key' : 'outcome_label'}

        """
        if self.est_keys is None:
            self.est_keys = {
                'treatment_label_key': 'treatment_label',
                'outcome_key': 'outcome',
                'treatment_pred_key': 'treatment',
                'outcome_label_key': 'outcome_label'}
        t_all = []
        q1_all = []
        q0_all = []
        g_all = []
        y_all = []
        for fold in foldNPZ:
            ld = np.load(fold)
            t_all.append(ld[self.est_keys['treatment_label_key']])
            q0_all.append(ld[self.est_keys['outcome_key']][:, 0])
            q1_all.append(ld[self.est_keys['outcome_key']][:, 1])
            y_all.append(ld[self.est_keys['outcome_label_key']])
            g_all.append(ld[self.est_keys['treatment_pred_key']][:, 1])
        t_all = np.array(list(itertools.chain(*t_all))).flatten()
        g_all = np.array(list(itertools.chain(*g_all))).flatten()
        q0_all = np.array(list(itertools.chain(*q0_all))).flatten()
        q1_all = np.array(list(itertools.chain(*q1_all))).flatten()
        y_all = np.array(list(itertools.chain(*y_all))).flatten()
        return q0_all, q1_all, y_all, g_all, t_all
