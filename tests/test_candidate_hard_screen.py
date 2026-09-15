import unittest
from copy import deepcopy

from scripts.candidate_hard_screen import split_plan, ensure_disjoint, select_training, POLICY


class HardScreenTests(unittest.TestCase):
    def row(self, i, category='reachable_correct', margin=20):
        return dict(query_index=i, path=str(i), category=category,
                    positive_minus_negative_margin=margin)

    def test_no_errors_does_not_select_easy_only(self):
        self.assertEqual(select_training([self.row(i) for i in range(10)])['selected'], [])

    def test_balanced_without_replacement(self):
        rows = [self.row(0, 'reachable_error', -2)] + [self.row(i, margin=i) for i in range(1, 12)]
        s = select_training(rows)
        self.assertEqual(s['errors'], [0]); self.assertEqual(s['near_correct'], [1])
        self.assertEqual(len(s['selected']), 3)
        self.assertEqual(len(set(s['selected'])), 3)
        self.assertEqual(s, select_training(rows))

    def test_unreachable_not_used_as_negative(self):
        rows = [self.row(0, 'unreachable'), self.row(1, 'all_positive'),
                self.row(2, 'reachable_error', -9), self.row(3), self.row(4)]
        self.assertEqual(set(select_training(rows)['selected']), {2, 3, 4})

    def test_insufficient_correct_not_oversampled(self):
        rows = [self.row(i, 'reachable_error', -1) for i in range(4)] + [self.row(4)]
        self.assertEqual(len(select_training(rows)['selected']), 5)

    def test_split_before_mining(self):
        groups = {}
        for city in ['A', 'B', 'C']:
            for i in range(1024):
                groups[f'{city}:{i}'] = [dict(city=city, date=[2000+j, 1],
                    path=f'{city}/{i}/{j}.jpg', panoid=f'{i}_{j}') for j in range(3)]
        plan = split_plan(groups, 768, 256, 1024)
        ensure_disjoint(plan)
        self.assertEqual(len(plan['train']['queries']), 768)
        self.assertEqual(len(plan['dev']['queries']), 256)
        self.assertEqual(len(plan['train']['database']), 1536)
        broken = deepcopy(plan)
        broken['dev']['queries'].append(broken['train']['queries'][0])
        with self.assertRaises(ValueError): ensure_disjoint(broken)

    def test_panorama_leakage_rejected(self):
        r = lambda label, path, pano: dict(label=label, path=path, city='A', panoid=pano)
        plan = dict(train=dict(queries=[r('a', 'q', 'same')], database=[r('a', 'd', 'd')]),
                    dev=dict(queries=[r('b', 'q2', 'same')], database=[r('b', 'd2', 'd2')]))
        with self.assertRaises(ValueError): ensure_disjoint(plan)

    def test_policy_uses_gsv_and_epoch_zero(self):
        self.assertIn('GSV', POLICY['selection'])
        self.assertIn('epoch zero', POLICY['selection'])
        self.assertEqual(POLICY['residual_bound'], 4)


if __name__ == '__main__': unittest.main()
