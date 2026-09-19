"""Small regression checks for stopping behavior and coordinate geometry."""
import unittest
import numpy as np
import torch
torch.set_num_threads(1)
from banking77 import policy, wilson
from screenspot import region_point
from clinc_validation import upper, metrics

class ResearchChecks(unittest.TestCase):
    def test_confidence_can_stop_at_each_checkpoint(self):
        probs={6:torch.tensor([[.95,.05],[.6,.4],[.6,.4],[.6,.4]]),
               12:torch.tensor([[.1,.9],[.95,.05],[.6,.4],[.6,.4]]),
               18:torch.tensor([[.1,.9],[.1,.9],[.95,.05],[.6,.4]]),
               24:torch.tensor([[.1,.9],[.1,.9],[.1,.9],[.1,.9]])}
        pred,depth=policy(probs,.9)
        self.assertEqual(depth.tolist(),[6,12,18,24])
        self.assertEqual(pred.tolist(),[0,0,0,1])
        pred,depth=policy(probs,1.01)
        self.assertEqual(depth.tolist(),[24]*4)
        self.assertEqual(pred.tolist(),[1]*4)

    def test_agreement_requires_a_preceding_head(self):
        probs={d:torch.tensor([[.95,.05]]) for d in [6,12,18,24]}
        _,depth=policy(probs,.9,agreement=True)
        self.assertEqual(depth.item(),12)

    def test_zero_observed_harm_is_not_zero_risk(self):
        self.assertGreater(wilson(0,32,1.645)[1],.01)
        self.assertLess(wilson(0,601,1.645)[1],.01)

    def test_exact_bound_has_known_zero_error_solution(self):
        self.assertEqual(upper(0,0),1.)
        self.assertEqual(upper(10,10),1.)
        self.assertAlmostEqual(upper(0,100),1-.05**(1/100))
        self.assertGreater(upper(0,50),.05)
        self.assertLess(upper(0,100),.05)

    def test_no_added_errors_can_still_mean_wrong_early_answers(self):
        labels=torch.tensor([0,1,0,1])
        prediction=torch.tensor([1,1,0,1])
        result=metrics(prediction,torch.tensor([12,12,24,24]),labels,prediction)
        self.assertEqual(result['harmful'],0)
        self.assertEqual(result['early_wrong'],1)
        self.assertEqual(result['early_count'],2)

    def test_region_center_combines_connected_patches(self):
        # Horizontal neighbours with weights .4/.3: weighted center x = 13/28.
        point=region_point(np.array([.4,.3,.02,.02]),2,2)
        self.assertAlmostEqual(point[0],13/28)
        self.assertAlmostEqual(point[1],.25)

    def test_row_boundary_does_not_join_distant_patches(self):
        # Array indices 2 and 3 are consecutive, but different ends of separate rows.
        point=region_point(np.array([0.,0.,.6,.4,0.,0.]),3,2)
        self.assertAlmostEqual(point[0],5/6)
        self.assertAlmostEqual(point[1],.25)

if __name__=='__main__':unittest.main()
