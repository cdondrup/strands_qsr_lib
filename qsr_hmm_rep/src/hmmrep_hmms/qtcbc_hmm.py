#!/usr/bin/env python

from hmmrep_hmms.qtc_hmm_abstractclass import QTCHMMAbstractclass
import numpy as np


class QTCBCHMM(QTCHMMAbstractclass):

    def __init__(self):
        super(QTCBCHMM, self).__init__()
        self.num_possible_states = 92

    def _create_transition_matrix(self, size, **kwargs):
        """Creates a Conditional Neighbourhood Diagram as a basis for the HMM"""

        qtc = []

        for i in xrange(1, 4):
            for j in xrange(1, 4):
                qtc.append([i-2, j-2, np.NaN, np.NaN])
        for i in xrange(1, 4):
            for j in xrange(1, 4):
                for k in xrange(1, 4):
                    for l in xrange(1, 4):
                        qtc.append([i-2, j-2, k-2, l-2])

        return super(QTCBCHMM, self)._create_transition_matrix(size=size, qtc=qtc)

    def _symbol_to_qsr(self, symbols):
        #TODO: Needs to be tested and altered to work with qtcb and qtcc
        multi = 2
        qtc = []
        q = 3**np.array(range(multi-1,-1,-1))
        for c in symbols:

            rc = np.array([c-1])
            print "rc", rc

            f = np.array([np.floor(rc[0]/q[0])])
            print "f", f
            r = np.fmod(rc[0],q[0])
            print "r", r

            for i in range(1, len(q)):
                print "i", i
                rc = np.append(rc, rc[i-1] - f[i-1] * q[i-1])
                print "rc", rc
                f = np.append(f, np.floor(rc[i]/q[i]))
                print "f", f
                r = np.append(r, np.fmod(rc[i],q[i]))
                print "r", r

            qtc.append(f-1)

        return self._qtc_num_to_str(qtc)