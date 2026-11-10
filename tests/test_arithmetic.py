import unittest
from local_model_app.arithmetic import calculate
class ArithmeticTests(unittest.TestCase):
    def test_conversions_and_product(self):
        self.assertEqual(calculate({'transmission_gw':'3000/1000','generation_gw':'(700+800)/1000','gw_miles':'30*(3000/1000)'}),
                         {'transmission_gw':3,'generation_gw':1.5,'gw_miles':90})
    def test_no_code_execution_or_unbounded_math(self):
        for expression in ["__import__('os')","2**9999999","1/0","float('nan')","1e200 * 2"]:
            with self.assertRaises((ValueError,ZeroDivisionError)):
                calculate({'x':expression})
