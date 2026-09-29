from collections import Counter
import pytest
from src.dsa_training import schedules


def plan():
    places=[str(i) for i in range(4096)]
    return dict(train_places=places,dev_places=['dev'],epochs=[dict(batches=[places[i:i+16] for i in range(0,4096,16)]) for _ in range(3)])


def test_full_coverage_matched_and_minority_hard():
    p=plan();s=schedules(p)
    assert s==schedules(p) and s[0]!=s[1]
    for epoch,batches in enumerate(s):
        assert len(batches)==320
        broad=[b for j,b in enumerate(batches) if j%5!=4]
        assert Counter(v for b in broad for v in b)==Counter(p['train_places'])
        assert batches[4::5]==p['epochs'][epoch]['batches'][:64]


def test_reject_leak_or_bad_batch():
    p=plan();p['dev_places']=['1']
    with pytest.raises(ValueError): schedules(p)
    p=plan();p['epochs'][0]['batches'][0][0]='dev'
    with pytest.raises(ValueError): schedules(p)
