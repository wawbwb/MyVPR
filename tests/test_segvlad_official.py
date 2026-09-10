from pathlib import Path
import pytest
from scripts.segvlad_official import vocabulary_assets


@pytest.mark.parametrize('mode,folder,suffix', [
    ('domain', 'indoor', '_r_fitted_pca_model_order3.pkl'),
    ('map', '17places', '_r_fitted_pca_model_order3_map.pkl'),
])
def test_vocabulary_pca_pair(mode, folder, suffix):
    cfg = {'domain_vlad_cluster': 'indoor', 'map_vlad_cluster': '17places'}
    exp = {'pca_model_pkl': '_r_fitted_pca_model_order3.pkl',
           'pca_model_pkl_map': '_r_fitted_pca_model_order3_map.pkl'}
    pca, centers, name = vocabulary_assets(Path('repo'), Path('data/17places'), cfg, exp, mode)
    assert pca.name == '17places'+suffix
    assert centers.parent.name == folder
    assert name == mode+'/'+folder
    assert 'dinoNV' not in str(pca)


def test_invalid_vocabulary_rejected():
    with pytest.raises(ValueError):
        vocabulary_assets(Path('repo'), Path('data'), {}, {}, 'unknown')
