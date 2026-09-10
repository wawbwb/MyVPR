from scripts.segvlad_vote_diagnostic import vote_rank


def test_cap_changes_duplicate_dominated_winner():
    raw, cap, stats = vote_rank([[0,0,1],[1,0,0]], [[.6,.6,.5],[.5,.1,.1]])
    assert raw[0] == 0
    assert cap[0] == 1
    assert stats[0]['vote_entries'] == 4
    assert stats[0]['distinct_query_segments'] == 2


def test_ties_follow_rank_major_first_seen_order():
    raw, cap, _ = vote_rank([[2,1]], [[.5,.5]])
    assert raw == cap == [2,1]


def test_distinct_query_segments_still_accumulate():
    _, _, stats = vote_rank([[0],[0]], [[.3],[.4]])
    assert abs(stats[0]['capped_vote']-.7) < 1e-9
    assert stats[0]['repeated_entries'] == 0
