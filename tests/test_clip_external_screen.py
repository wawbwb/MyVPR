import unittest
import numpy as np
from scripts.clip_external_screen import external_map, DYNAMIC, EGO, STATIC, controls, VARIANTS


class ExternalScreenTest(unittest.TestCase):
    def scores(self):
        return np.zeros((16, len(DYNAMIC)+len(EGO)+len(STATIC)))

    def test_ego_has_no_effect_and_ties_do_not_suppress(self):
        c=self.scores()
        np.testing.assert_array_equal(external_map(c),0)
        c[:,0]=.1
        expected=external_map(c)
        c[:,len(DYNAMIC):len(DYNAMIC)+len(EGO)]=100
        np.testing.assert_array_equal(external_map(c),expected)
        self.assertGreater(expected.min(),0)

    def test_static_winner_and_local_support(self):
        c=self.scores(); c[:,len(DYNAMIC)+len(EGO)]=.1
        np.testing.assert_array_equal(external_map(c),0)
        c[0,0]=.2
        m=external_map(c)
        self.assertTrue((m[:8,:8]>0).all())
        self.assertFalse(m[8:,:].any())
        self.assertFalse(m[:,8:].any())

    def test_shuffle_preserves_exact_row_distribution(self):
        c=self.scores(); c[:,0]=np.linspace(0,.2,16)
        m=external_map(c); shuffled=controls(m,'fixed/query.jpg')['shuffle']
        np.testing.assert_array_equal(np.sort(m,axis=1),np.sort(shuffled,axis=1))
        np.testing.assert_array_equal(shuffled,controls(m,'fixed/query.jpg')['shuffle'])
        self.assertEqual(VARIANTS,['baseline','aligned_input','shuffle_input'])

    def test_rejects_bad_cosines(self):
        c=self.scores(); c[0,0]=np.nan
        with self.assertRaises(ValueError): external_map(c)
        with self.assertRaises(ValueError): external_map(np.zeros((1,17)))


if __name__=='__main__':
    unittest.main()
