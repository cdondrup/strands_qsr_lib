#!/usr/bin/env python
from __future__ import print_function, division
import sys
from abstractclass_unittest_dyadic import Abstractclass_Unittest_Dyadic


class RCC4_Test(Abstractclass_Unittest_Dyadic):
    def __init__(self, *args):
        super(RCC4_Test, self).__init__(*args)
        self._unique_id = "rcc4"
        self.__params = {"quantisation_factor": 2.0}
        self.__custom = {self._unique_id: {"qsrs_for": [("o1", "o2")],
                                           "quantisation_factor": 2.0}}

    def test_defaults(self):
        self.assertItemsEqual(*self.defaults("data1", "data1_rcc4_defaults.txt"))

    def test_qsrs_for_global_namespace(self):
        self.assertItemsEqual(*self.qsrs_for_global_namespace("data1", "data1_rcc4_qsrs_for_global_namespace.txt"))

    def test_qsrs_for_qsr_namespace(self):
        self.assertItemsEqual(*self.qsrs_for_qsr_namespace("data1", "data1_rcc4_qsrs_for_qsr_namespace.txt"))
        # precedes "for_all_qsrs" namespace
        self.assertItemsEqual(*self.qsrs_for_qsr_namespace_over_global_namespace("data1",
                                                                                 "data1_rcc4_qsrs_for_qsr_namespace.txt"))

    def test_q_factor(self):
        self.assertItemsEqual(*self.q_factor("data1", "data1_rcc4_q_factor_2p0.txt",
                                             self.__params["quantisation_factor"]))
        q_factor_results, defaults_results = self.q_factor_data_notequal_defaults("data1_rcc4_q_factor_2p0.txt",
                                                                                  "data1_rcc4_defaults.txt")
        self.assertFalse(q_factor_results == defaults_results)

    def test_with_bounding_boxes(self):
        self.assertItemsEqual(*self.defaults("data2", "data2_rcc4_defaults.txt"))

    def test_without_bounding_boxes(self):
        self.assertItemsEqual(*self.defaults("data3", "data3_rcc4_defaults.txt"))

    def test_floats(self):
        self.assertItemsEqual(*self.defaults("data4", "data4_rcc4_defaults.txt"))

    def test_custom(self):
        self.assertItemsEqual(*self.custom("data1", "data1_rcc4_custom.txt", self.__custom))


if __name__ == '__main__':
    import rosunit
    rosunit.unitrun("qsr_lib", "rcc4_test", RCC4_Test, sys.argv)
