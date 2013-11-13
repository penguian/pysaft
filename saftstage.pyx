"""
 * saftstage.pyx
 * Copyright (C) 2013  Paul LEOPARDI
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import saftsparse
import saftstats

def setup_dna_alphabet():
    """
    Determine alphabet size and letter frequency.
    """
    alphabet = saftsparse.dna_alphabet
    alpha = len(alphabet)
    alpha_freq = np.ones(alpha) / alpha
    return alphabet, alpha, alpha_freq

def calculate_d2_statistic(inp_freq, dat_freq):
    """
    Calculate d2 statistic.
    """
    d2_vals = np.asarray((inp_freq.T * dat_freq).todense())[0]
    return d2_vals

def calculate_means_vars(args,
                         context,
                         inp_freq,
                         inp_size,
                         dat_freq,
                         dat_size):
    """
    Calculate theroretical means and variances.
    """
    dat_len = dat_freq.shape[1]
    nbr_inp_words = inp_size + args.wordsize - 1

    d2_means = np.array([
        saftstats.mean(context,
                       nbr_inp_words,
                       dat_size[j] + args.wordsize- 1)
        for j in xrange(dat_len)])
    d2_vars  = np.array([
        saftstats.var(context,
                      nbr_inp_words,
                      dat_size[j] + args.wordsize- 1)
        for j in xrange(dat_len)])
    return d2_means, d2_vars

def print_hit(stat_name, description, stat_val, adj_pval, pval):
    print "  Hit:", description,
    print stat_name+":", "{:d}".format(long(stat_val)),
    print "adj.p.val:", "{:11.5e}".format(adj_pval),
    print "p.val:", "{:11.5e}".format(pval)

def print_hit_d2(description, stat_val, adj_pval, pval):
    print_hit("D2", description, stat_val, adj_pval, pval)

def print_no_hit():
    print "No hit found"

def print_query_results_d2(args,
                           inp_desc,
                           dat_desc,
                           d2_vals,
                           d2_pvals,
                           dat_len,
                           nbr_pvals,
                           dat_desc_index=lambda j:j):
    print "Query:", inp_desc,
    print "program: saftn word size:", args.wordsize
    jsorted = np.argsort(d2_pvals[:nbr_pvals])
    d2_adj_pvals = saftstats.BH_array(d2_pvals[jsorted], dat_len)
    nbr_pvals = min(args.showmax, nbr_pvals)
    jrange = [j for j in xrange(nbr_pvals)
              if d2_adj_pvals[j] < args.pmax]
    if len(jrange) > 0:
        for j in jrange:
            js = jsorted[j]
            jg = dat_desc_index(js)
            print_hit_d2(dat_desc[jg], d2_vals[js], d2_adj_pvals[j], d2_pvals[js])
    else:
        print_no_hit()
