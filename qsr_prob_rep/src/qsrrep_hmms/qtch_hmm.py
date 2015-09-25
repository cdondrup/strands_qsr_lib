# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:33:38 2015

@author: cdondrup
"""

from qsrrep_hmms.qtc_hmm_abstractclass import QTCHMMAbstractclass
import numpy as np


class QTCHHMM(QTCHMMAbstractclass):

    def __init__(self):
        super(QTCHHMM, self).__init__()
        self.num_possible_states = 110 # Setting number of possible states: 3^3 + 3^4 + start and end


    def _create_transition_matrix(self, size, **kwargs):
        """Method for the creation of the transition probability matrix. Creates
        a uniformly distributed matrix. Please override if special behaviour
        is necessary.

        :return: uniform SIZExSIZE transition matrix as a numpy array
        """

        trans = np.ones([size,size])
        return trans/trans.sum(axis=1)

    def _symbol_to_qsr(self, symbols):
        """Transforming alphabet symbols to QTCBC states.

        :param symbols: A list of symbols

        :return: The list of corresponding qtc symbols
        """

        ret = []
        for s in symbols:
            qtc = []
            for c in s[1:-1]:
                if c <= 27:
                    qtc.append(self.symbol_to_qsr(c, 3**np.array(range(1,-1,-1))))
                else:
                    qtc.append(self.symbol_to_qsr(c-27, 3**np.array(range(3,-1,-1))))

            ret.append(self._qtc_num_to_str(qtc))

        return ret

    def symbol_to_qsr(self, symbol, multiplier):
        rc = np.array([symbol-1])
        f = np.array([np.floor(rc[0]/multiplier[0])])

        for i in range(1, len(multiplier)):
            rc = np.append(rc, rc[i-1] - f[i-1] * multiplier[i-1])
            f = np.append(f, np.floor(rc[i]/multiplier[i]))

        return f-1

    def _qsr_to_symbol(self, qsr_data):
        """Transforms a qtc state chain to a list of numbers

        :param qsr_data: The list of lists of qtc strings or numpy array states
            E.g.: [['++++','+++0','+++-,]] or [[[1,1,1,1],[1,1,1,0],[1,1,1,-1]]]

        :return: A lists of lists of alphabet symbols corresponding to the given state chains
        """
        qsr_data = np.array(qsr_data)
        state_rep = []
        for idx, element in enumerate(qsr_data):
            if all(isinstance(x, str) or isinstance(x, unicode) for x in element):
                element = self._qtc_str_to_num(element) # check if content is string instead of numbers and convert
            element = np.array(element)
            try:
                d = element.shape[1]
            except IndexError: # Not a list of lists of lists
                return self._qsr_to_symbol([qsr_data])
            state_num = np.array([0]) # Start symbol
            #Not ellegant at all but necessary due to the nan values and the different multipliers for qtcb and qtcc
            for x in element:
                x = x[~np.isnan(x)]
                d = x.shape[0]
                mult = 3**np.arange(d-1, -1, -1)
                num = ((x + 1)*mult).sum() + 1
                state_num = np.append(
                    state_num,
                    num if d == 3 else num + 27 # Adding a 27 when the state is qtcc
                )
            state_num = np.append(state_num, self.num_possible_states-1) # End symbol
            state_char = ''
            for n in state_num:
                state_char += chr(int(n)+32)
            state_rep.append(state_num.tolist())

        return state_rep