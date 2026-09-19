"""Checks error accounting that can otherwise hide unsafe shortcuts."""
import unittest
import torch
torch.set_num_threads(1)
from chat600 import measures,select,upper

class ChatChecks(unittest.TestCase):
    def test_equal_accuracy_can_hide_a_new_toxic_miss(self):
        y=torch.tensor([1,0,1,0]);full=torch.tensor([1,1,1,0]);early=torch.tensor([0,0,1,0])
        m=measures(early,y,torch.tensor([6,6,24,24]),full)
        self.assertEqual(m['correct'],3)
        self.assertEqual(m['added_errors'],1)
        self.assertEqual(m['corrected_errors'],1)
        self.assertEqual(m['additional_missed_toxic'],1)
        self.assertAlmostEqual(m['added_miss_upper95'],upper(1,2))

    def test_always_safe_has_zero_toxic_recall(self):
        m=measures(torch.zeros(100,dtype=torch.long),torch.tensor([1]*5+[0]*95))
        self.assertEqual(m['accuracy'],.95)
        self.assertEqual(m['toxic_recall'],0)
        self.assertEqual(m['missed_toxic'],5)

    def test_agreement_and_disabled_gate(self):
        p={d:torch.tensor([[.01,.99]]) for d in [6,12,18,24]}
        _,depth=select(p,.9,True,6);self.assertEqual(depth.item(),12)
        _,depth=select(p,1.01,False,6);self.assertEqual(depth.item(),24)

if __name__=='__main__':unittest.main()
