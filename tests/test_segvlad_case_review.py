from scripts.segvlad_case_review import window_info


def test_window_boundaries():
    assert window_info(243, 258)['official_hit']
    assert not window_info(243, 259)['official_hit']
    assert window_info(243, 228)['distance_to_window_boundary'] == 0
    assert window_info(243, 227)['offset'] == -16
