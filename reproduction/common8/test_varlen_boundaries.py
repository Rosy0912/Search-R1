"""Exercise the installed HF packed dispatch on CPU, not a GPU speed claim."""
import unittest
from unittest.mock import patch
import torch
import transformers.modeling_flash_attention_utils as fa

class PackedBoundaryTests(unittest.TestCase):
    def test_real_hf_dispatch_separates_packed_examples(self):
        q=torch.randn(1,7,2,4);positions=torch.tensor([[0,1,2,0,1,2,3]])
        calls=[]
        def isolated(query,key,value,**kwargs):
            boundaries=kwargs['cu_seqlens_q'].tolist();calls.append(boundaries)
            out=query.clone()
            for start,end in zip(boundaries,boundaries[1:]):
                out[start:end]=value[start:end].mean(0)
            return out
        with patch.object(fa,'_flash_supports_window_size',False,create=True),patch.object(fa,'flash_attn_varlen_func',isolated,create=True),patch.object(fa,'flash_attn_func',side_effect=AssertionError('Packed path lost boundaries'),create=True):
            a=fa._flash_attention_forward(q,q,q,None,7,True,position_ids=positions)
            changed=q.clone();changed[:,3:]+=100
            b=fa._flash_attention_forward(changed,changed,changed,None,7,True,position_ids=positions)
        self.assertEqual(calls,[[0,3,7],[0,3,7]])
        self.assertTrue(torch.equal(a[:,:3],b[:,:3]))
        self.assertFalse(torch.equal(a[:,3:],b[:,3:]))
