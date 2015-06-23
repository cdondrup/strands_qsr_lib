#!/usr/bin/env python

from abc import ABCMeta
import numpy as np
from hmmrep_hmms.hmm_abstractclass import HMMAbstractclass


class QTCHMMAbstractclass(HMMAbstractclass):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(QTCHMMAbstractclass, self).__init__()

    def _create_transition_matrix(self, size, **kwargs):
        """Creates a Conditional Neighbourhood Diagram as a basis for the HMM"""

        qtc = np.array(kwargs["qtc"])
        #np.savetxt('/home/cdondrup/qtc.csv', qtc, delimiter=',', fmt='%1f')

        trans = np.zeros((size, size))
        for i1 in xrange(qtc.shape[0]):
            for i2 in xrange(i1+1, qtc.shape[0]):
                trans[i1+1, i2+1] = np.nanmax(np.absolute(qtc[i1]-qtc[i2])) != 2
                if trans[i1+1, i2+1] == 1:
                    for j1 in xrange(qtc.shape[1]-1):
                        for j2 in xrange(j1+1, qtc.shape[1]):
                            if sum(np.absolute(qtc[i1, [j1, j2]])) == 1 \
                                    and sum(np.absolute(qtc[i2, [j1, j2]])) == 1:
                                if np.nanmax(np.absolute(qtc[i1, [j1, j2]]-qtc[i2, [j1, j2]])) > 0 \
                                        and sum(qtc[i1, [j1, j2]]-qtc[i2, [j1,j2]]) != 1:
                                    trans[i1+1, i2+1] = 5
                                    break
                        if trans[i1+1, i2+1] != 1:
                            break
                trans[i2+1, i1+1] = trans[i1+1, i2+1]

        trans[trans != 1] = 0
        #np.savetxt('/home/cdondrup/trans.csv', np.rint(trans).astype(int), delimiter=',', fmt='%i')
        trans[trans == 0] = 0.00001
        trans[0] = 1
        trans[:, 0] = 0
        trans[:, -1] = 1
        trans[0, -1] = 0
        trans[-1] = 0
        trans += np.dot(np.eye(size), 0.00001)
        trans[0, 0] = 0

        trans = trans / trans.sum(axis=1).reshape(-1, 1)
        #np.savetxt('/home/cdondrup/trans.csv', trans, delimiter=',')

        return trans

    def _create_emission_matrix(self, size, **kwargs):
        emi = np.eye(size)
        emi[emi == 0] = 0.0001

        return emi


    def _qsr_to_symbol(self, qsr_data):
        """Transforms a qtc state to a number"""
        qsr_data = np.array(qsr_data)
        state_rep = []
        for idx, element in enumerate(qsr_data):
            element = np.array(element)
            d = element.shape[1]
            mult = 3**np.arange(d-1, -1, -1)
            state_num = np.append(
                0,
                ((element + 1)*np.tile(mult, (element.shape[0], 1))).sum(axis=1) + 1
            )
            state_num = np.append(state_num, 82)
            state_char = ''
            for n in state_num:
                state_char += chr(int(n)+32)
            state_rep.append(state_num.tolist())

        return state_rep

    def _qtc_num_to_str(self, qtc_num_list):
        qtc_str = []
        for elem in qtc_num_list:
            s = ''
            for num in elem:
                if num == 0:
                    s +='0'
                elif num == 1:
                    s +='+'
                elif num == -1:
                    s +='-'
            qtc_str.append(s)
        return qtc_str