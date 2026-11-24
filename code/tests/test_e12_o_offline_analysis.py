from scripts.e12o_o2_offline_analysis import candidate_ids


def test_e12o_offline_analysis_supports_bright_and_garage_candidate_schemas():
    bright = {"candidate_ids": ["b1", "b2"]}
    garage = {"candidate_order": ["g1", "g2"]}
    assert candidate_ids(bright) == ["b1", "b2"]
    assert candidate_ids(garage) == ["g1", "g2"]
