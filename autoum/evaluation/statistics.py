import itertools as it

import numpy as np
import pandas as pd
from scipy.stats import f, rankdata, studentized_range


class Statistics:
    """
    Class for statistical comparison of classifiers over multiple data sets

    Documentation only! Please aware that we need a scipy version > 1.4.1 but causaml and econml require a causaml version == 1.4.1
    """

    @staticmethod
    def calculate_ranks(a, y_col, group_col, block_col, melted, sort):
        """
        Given an np.ndarray with size (k, n), calculate the ranks for each data set and algorithm

        See Statistics.nemenyi_srs for parameter description
        """
        from scikit_posthocs import __convert_to_block_df

        x, _y_col, _group_col, _block_col = __convert_to_block_df(a, y_col, group_col, block_col, melted)
        x = x.sort_values(by=[_group_col, _block_col], ascending=True) if sort else x
        x.dropna(inplace=True)

        groups = x[_group_col].unique()
        k = groups.size
        n = x[_block_col].unique().size

        # Calculate ranks
        x['mat'] = x.groupby(_block_col)[_y_col].rank()
        R = x.groupby(_group_col)['mat'].mean()
        vs = np.zeros((k, k))
        combs = it.combinations(range(k), 2)

        return combs, R, x, groups, vs, k, n

    @staticmethod
    def nemenyi_srs(a, alpha, columns, y_col=None, block_col=None, group_col=None, melted=False, sort=False):
        """
        Post-hoc Nemenyi test with Studentized Rank Statistics (own implementation):

        The performance of two classifiers i and j is significant different if the corresponding average ranks R_i and R_j differ by
        at least the critical difference:

        CD = q_a*\sqrt{k(k+1)/6N}

        where critical values q_a are based on the Studentized range statistic divided by \sqrt{2}

        :param a: array_like or pandas DataFrame object. An array, any object exposing the array interface or a pandas DataFrame.
        :param alpha: Alpha value
        :param columns: Name of the columns
        :param y_col: Must be specified if `a` is a pandas DataFrame object. Name of the column that contains y data.
        :param block_col: Must be specified if `a` is a pandas DataFrame object. Name of the column that contains blocking factor values.
        :param group_col: Must be specified if `a` is a pandas DataFrame object. Name of the column that contains treatment (group) factor values.
        :param melted: Specifies if data are given as melted columns "y", "blocks", and "groups". Default: False
        :param sort: If True, sort data by block and group columns. Default: False
        :return: DataFrame containing 0 and 1, whereby 1 refers to a pairwise comparison with a significant difference with alpha

        """
        combs, R, x, groups, vs, k, n = Statistics.calculate_ranks(a, y_col, group_col, block_col, melted, sort)

        tri_upper = np.triu_indices(vs.shape[0], 1)
        tri_lower = np.tril_indices(vs.shape[0], -1)
        vs[:, :] = 0
        significances = np.full((vs.shape[0], vs.shape[1]), True)

        # Calculate pairwise difference in average ranks over the data sets
        for i, j in combs:
            vs[i, j] = np.abs(R[groups[i]] - R[groups[j]])

        # Get the critical value
        cv = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)

        # Calculate the critical difference
        cd = cv * np.sqrt((k * (k + 1)) / (6 * n))

        # Return 1 if the difference in ranks is equal or higher than the critical difference
        significances[tri_upper] = vs[tri_upper] >= cd
        significances[tri_lower] = np.transpose(vs)[tri_lower] >= cd
        significances = significances.astype(int)
        np.fill_diagonal(significances, 0)

        df = pd.DataFrame(significances, index=groups, columns=groups)

        df.set_axis(columns, inplace=True)
        df.set_axis(columns, axis=1, inplace=True)

        return df

    @staticmethod
    def calculate_friedman_test(*args):
        """
        Friedman test (non-parametric) with N (# of data sets) and k (# of algorithms) are big (e.g., N>10 and k>5). In our case it is usally k>10 and N>5.

        Null hypothesis: There are no differences in performance

        1. Calculate the ranks for each data set (rank the algorithm per data sets)
        2. Calculate the chi-squared approximation of the Friedman test
        3. Calculate the number of degrees of freedom (k-1)
        4. Calculate the F-Distribution
        4. Check the Table of critical value for the F distribution using a significance value alpha and the degrees of freedom
        5. Reject the null hypothesis if the calculated F-Distribution is equal to or greater than the table critical chi-squared value at the prespecified level of significance

        :param args: (list of array, each containing the performance on an algorithm over the n data sets)
        :return: The p-value
        """
        k = len(args)
        if k < 3:
            raise ValueError('Less than 3 levels.  Friedman test not appropriate.')

        n = len(args[0])
        for i in range(1, k):
            if len(args[i]) != n:
                raise ValueError('Unequal N in friedmanchisquare.  Aborting.')

        # Rank data
        data = np.vstack(args).T
        data = data.astype(float)
        for i in range(len(data)):
            data[i] = k + 1 - rankdata(data[i])

        # Calculate the sum of average ranks of all algorithms
        ssbn = np.sum((data.sum(axis=0) / n) ** 2)

        # Approximate the chi-sqaure of the Friedman test
        chisq = ((12.0 * n) / (k * (k + 1))) * (ssbn - ((k * (k + 1) ** 2) / 4))

        # Calculate the F-distribution
        f_dist = (n - 1) * chisq / (n * (k - 1) - chisq)

        # Calculate the p-value
        p = f.sf(f_dist, k - 1, (k - 1) * (n - 1))

        return round(p, 4)