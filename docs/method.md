# Bounded preference model

The current browser demo learns logistic regression on drawing parameters. The renderer is procedural SVG, not a pretrained image generator. Canonical implementations: `scripts/preference-core.js` and `experiments/preference.py`. The larger neural example is illustrative.

For z in [-1,1]^d, the predicted score is sigmoid(w·z+b). Fit binary cross entropy plus L2 weight regularization using 260 full-batch steps, learning rate 0.5 and regularization 0.02. Reset weights before fitting so results depend on ratings rather than previous optimizer state. Scores are not calibrated probabilities of a person's future preference.

## Maximization

The sigmoid is monotone. Each nonzero coordinate of the maximum is sign(w_i); zero-weight coordinates are arbitrary. For the smaller box [-t,t]^d, use t·sign(w_i). The slider changes the feasible box. It is not a proven realism control, and the maximum is not necessarily a score of 1.

The positive region of a linear model intersected with a box is convex. Random restarts, perturbations and coordinate search do not create disconnected preference modes. A richer feature map, nonlinear head or mixture is needed.

## Minimum-change edit

Minimize ||z-r||² subject to w·z+b >= log(p/(1-p)) and -1 <= z_i <= 1. Return r when its score already meets the target. If b+sum(abs(w)) is below the target logit, report `unreachable` and return a score-maximizing point, preserving zero-weight coordinates.

For a feasible target, the KKT solution has the form z_i = clip(r_i + lambda*w_i, -1, 1), with lambda >= 0. The score is monotone in lambda, so bracket and bisect. The implementation normalizes the search direction to avoid an arbitrary small-weight cutoff. Clipping an unconstrained projection once does not generally solve the bounded problem: clipping changes the active coordinates.

Numerical checks compare 80 random cases with SciPy SLSQP and share fixtures with the JavaScript core. Small latent L2 distance does not itself establish small perceptual change.

## Evaluation and sessions

Training fit uses the same ratings used to train. Blind evaluation draws uniformly, hides the score, and records predictions without retraining; evaluation is cleared when training changes. It remains an exploratory sequential session, not a blinded controlled study. Exported sessions contain parameters and ratings; imports validate bounds and recompute the model locally. Decorative rendering uses a separate random stream from candidate selection.

All current demo computation stays in the browser. This does not establish that exported features or future services preserve privacy.
