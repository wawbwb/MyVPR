from src.query_relation_training import protocol_schedule
import pytest


def plan():
    places = [str(i) for i in range(4096)]
    batches = [places[j:j+16] for j in range(0, 4096, 16)]
    return dict(train_places=places, dev_places=['dev'], epochs=[dict(batches=batches) for _ in range(3)])


def test_matched_schedule():
    p = plan()
    mixed = protocol_schedule(p, 'mixed')
    broad = protocol_schedule(p, 'broad_matched')
    assert broad == protocol_schedule(p, 'broad_matched')
    assert len(broad) == len(mixed) == 3
    for x, y in zip(mixed, broad):
        assert len(x) == len(y) == 320
        assert all(len(b) == len(set(b)) == 16 for b in y)
        assert all(set(b) <= set(p['train_places']) for b in y)
        assert [b for i,b in enumerate(x) if i%5 != 4] == [b for i,b in enumerate(y) if i%5 != 4]
        assert len(set(v for i,b in enumerate(y) if i%5 == 4 for v in b)) == 1024
    assert p == plan(), 'Do not mutate the historical plan'
    with pytest.raises(ValueError): protocol_schedule(p, 'unknown')
