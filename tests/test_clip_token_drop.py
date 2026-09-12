import importlib.util
import types
import unittest
import numpy as np
from src.models.clip_token_drop import drop_mask

HAS_TORCH=importlib.util.find_spec('torch') is not None


class MaskTests(unittest.TestCase):
    def test_threshold_and_fallback(self):
        f=np.zeros((20,20),np.float16); f[:3]=.75
        mask,fb=drop_mask(f,'image','aligned')
        self.assertEqual(mask.sum(),60); self.assertFalse(fb)
        mask,fb=drop_mask(np.ones((20,20)),'image','aligned')
        self.assertFalse(mask.any()); self.assertTrue(fb)

    def test_none_and_shuffle(self):
        f=np.zeros((20,20)); f[3:10,2:8]=1
        a,_=drop_mask(f,'image','aligned'); b,_=drop_mask(f,'image','shuffled')
        np.testing.assert_array_equal(a.sum(1),b.sum(1))
        self.assertFalse(np.array_equal(a,b))
        np.testing.assert_array_equal(b,drop_mask(f,'image','shuffled')[0])
        self.assertFalse(drop_mask(f,'image','none')[0].any())

    def test_invalid(self):
        for f in [np.zeros((10,10)),np.full((20,20),np.nan),np.full((20,20),1.1)]:
            with self.assertRaises(ValueError): drop_mask(f,'image','aligned')


@unittest.skipUnless(HAS_TORCH,'PyTorch is required: run these tests on the VPR server')
class TorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        from src.models.aggregators.boq import BoQ
        torch.manual_seed(123); torch.set_num_threads(1)
        self.torch=torch
        self.agg=BoQ(in_channels=8,proj_channels=64,num_queries=4,num_layers=2,row_dim=4).eval()

    def test_all_keep_matches_original(self):
        from src.models.clip_token_drop import aggregate
        t=self.torch; x=t.randn(2,8,4,4)
        t.testing.assert_close(aggregate(self.agg,x,t.zeros(2,4,4,dtype=t.bool)),self.agg(x)[0],rtol=0,atol=0)

    def test_padding_matches_individual_and_gradients(self):
        from src.models.clip_token_drop import compact_tokens,encode_compact
        t=self.torch; x=t.randn(2,12,64,requires_grad=True)
        drop=t.zeros(2,12,dtype=t.bool);drop[0,2:9]=True;drop[1,4:7]=True
        compact,padding=compact_tokens(x,drop)
        batched=encode_compact(self.agg,compact,padding)
        individual=t.cat([encode_compact(self.agg,x[i:i+1,~drop[i]],t.zeros(1,int((~drop[i]).sum()),dtype=t.bool)) for i in range(2)])
        t.testing.assert_close(batched,individual,rtol=2e-5,atol=2e-6)
        batched[:,0].sum().backward()
        self.assertEqual(x.grad[drop].abs().max().item(),0)
        self.assertGreater(x.grad[~drop].abs().sum().item(),0)

    def test_dropped_projected_values_cannot_change_result(self):
        from src.models.clip_token_drop import compact_tokens,encode_compact
        t=self.torch; x=t.randn(2,12,64); drop=t.zeros(2,12,dtype=t.bool);drop[:,::3]=True
        z,p=compact_tokens(x,drop); y=encode_compact(self.agg,z,p)
        x[drop]=10000*t.randn_like(x[drop]);z2,p2=compact_tokens(x,drop)
        t.testing.assert_close(y,encode_compact(self.agg,z2,p2),rtol=0,atol=0)

    def test_qq_matches_attention_with_tied_qk(self):
        from src.models.clip_token_drop import qq_attention_only
        t=self.torch; attn=t.nn.MultiheadAttention(16,4,batch_first=True)
        block=types.SimpleNamespace(attn=attn,ln_1=t.nn.LayerNorm(16))
        x=t.randn(2,7,16); actual=qq_attention_only(block,x)
        with t.no_grad():
            attn.in_proj_weight[16:32].copy_(attn.in_proj_weight[:16])
            attn.in_proj_bias[16:32].copy_(attn.in_proj_bias[:16])
        n=block.ln_1(x); expected=attn(n,n,n,need_weights=False)[0]
        t.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)

    def test_empty_compaction_rejected(self):
        from src.models.clip_token_drop import compact_tokens
        with self.assertRaises(ValueError): compact_tokens(self.torch.zeros(1,4,8),self.torch.ones(1,4,dtype=self.torch.bool))


if __name__=='__main__': unittest.main()
