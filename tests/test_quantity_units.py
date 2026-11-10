import unittest

from local_model_app.quantity_units import source_unit, convert


class QuantityUnitsTests(unittest.TestCase):
    def test_source_column_and_reported_metadata(self):
        self.assertEqual(source_unit({},'/structuredContent/numeric_columns/InstalledMW/sum')['unit'],'MW')
        self.assertEqual(source_unit({},'/structuredContent/numeric_columns/RatingMVA/sum')['unit'],'MVA')
        self.assertEqual(source_unit({},'/structuredContent/numeric_columns/Capacity/sum')['unit'],'unknown')
        self.assertEqual(source_unit({'structuredContent':{'column_units':{'Capacity':'MW'}}},
            '/structuredContent/numeric_columns/Capacity/sum')['unit'],'MW')

    def test_conversion_and_power_factor_are_explicit(self):
        self.assertEqual(convert(1500,'MW','GW')['value'],1.5)
        self.assertEqual(convert(3000,'MVA','GW',1)['value'],3)
        self.assertEqual(convert(3000,'MVA','GW',0.9)['value'],2.7)
        for source,target,pf in [('unknown','GW',None),('MVA','GW',None),('miles','GW',None),('MW','GW',1)]:
            with self.assertRaises(ValueError): convert(1500,source,target,pf)
        with self.assertRaises(ValueError): convert(float('inf'),'MW','GW')


if __name__=='__main__': unittest.main()
