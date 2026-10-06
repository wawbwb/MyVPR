import torch
import pytest
from src.losses.vpr_losses import VPRLossFunction
from src.query_loss_audit import paired_losses


def test_identity_and_shape():
    torch.manual_seed(42)
    x=torch.nn.functional.normalize(torch.randn(8,16),dim=1)
    labels=torch.arange(4).repeat_interleave(2)
    r=paired_losses(VPRLossFunction(),x,x,labels)
    assert r['ru_loss']==r['trained_loss']==r['trained_ru_pairs_loss']
    assert r['descriptor_l2']==0
    assert r['ru_positive_pairs']==r['trained_positive_pairs']
    with pytest.raises(ValueError): paired_losses(VPRLossFunction(),x,x[:4],labels)


def test_easy_batch_zero_pairs():
    x=torch.eye(4).repeat_interleave(2,0)
    r=paired_losses(VPRLossFunction(),x,x,torch.arange(4).repeat_interleave(2))
    assert r['ru_loss']==r['trained_loss']==0
