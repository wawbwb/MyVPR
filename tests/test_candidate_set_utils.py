import unittest
import numpy as np
from src.candidate_set_utils import choose_views,make_plan,summary


class ProtocolTests(unittest.TestCase):
    def test_distinct_panoramas(self):
        rows=[{'path':str(i),'date':[2020,i],'panoid':str(i//2)} for i in range(6)]
        q,db=choose_views(rows)
        self.assertEqual(q['panoid'],'0');self.assertEqual({r['panoid'] for r in db},{'1','2'})
        self.assertIsNone(choose_views(rows[:4]))

    def test_city_split_full_development_database(self):
        groups={}
        for city in ['a','b','c']:
            for i in range(1100):
                groups[f'{city}:{i}']=[{'city':city,'path':f'{city}/{i}/{j}','date':[2020,j],'panoid':f'{i}-{j}'} for j in range(3)]
        plan=make_plan(groups,train_queries=64,eval_queries=32,train_places=1024)
        sets=[{r['city'] for r in plan[s]['database']} for s in ['train','dev','test']]
        self.assertTrue(all(not a&b for i,a in enumerate(sets) for b in sets[i+1:]))
        self.assertEqual(len(plan['dev']['database']),2200)
        for split in ['train','dev','test']:
            q={(r['city'],r['panoid']) for r in plan[split]['queries']}
            d={(r['city'],r['panoid']) for r in plan[split]['database']}
            self.assertFalse(q&d)
        self.assertEqual(plan,make_plan(dict(reversed(list(groups.items()))),64,32,1024))

    def test_unreachable_not_dropped_and_multi_positive(self):
        labels=np.array([[True,True,False],[False,False,False],[False,True,False]])
        base=np.array([[3,2,1],[3,2,1],[3,2,1]],float)
        scores=np.array([[1,2,3],[1,2,3],[1,3,2]],float)
        s=summary(scores,labels,base)
        self.assertEqual((s['queries'],s['reachable'],s['correct']),(3,2,1))
        self.assertEqual(s['corrected'],[2]);self.assertEqual(s['regressed'],[0])
        self.assertEqual(s['recall']['5'],2/3)


if __name__=='__main__':unittest.main()
