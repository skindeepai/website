# Maze actions from a small Qwen model

This experiment tests whether a frozen Qwen model can choose a useful action from a small ASCII maze, and whether trained intermediate classifiers can return that action before all 24 transformer blocks execute. It uses synthetic, fully observed, static 4-by-4 mazes. It is not Pac-Man: there are no moving enemies, scores, hidden state or visual perception.

**Both learned policies reached zero of ten episode goals. BFS reached all ten.** The early candidate sometimes returned actions with fewer executed blocks, but this run does not demonstrate a useful maze navigator or a quality-preserving practical speedup. Its calibration guard also failed.

## Measured outcomes

| Policy | Optimal moves on 128 held-out states | Legal moves on those states | Episode goals reached | Episode legal actions |
| --- | --- | --- | --- | --- |
| Trained full-depth head | 83/128 | 110/128 | 0/10 | 85/200 |
| Early candidate | 87/128 | 112/128 | 0/10 | 85/200 |
| BFS control | 128/128 | 128/128 | 10/10 | 59/59 |

All BFS episodes followed shortest paths, requiring 4 to 8 actions. The learned policies each used all 20 allowed actions on every episode. Each accumulated 115 illegal actions; repeated collisions were retained rather than corrected by an oracle.

The state-test candidate stopped at layer 6 on 29 states, layer 12 on six, layer 18 on eight, and full depth on 85. These were actual stopped forward passes, matching the cached readout predictions, with **20.90% of blocks skipped** over the 128 state tests. The candidate introduced two errors the full head avoided and corrected six full-head errors. Similar aggregate accuracy conceals these changed answers.

Closed-loop observations gave a different exit distribution: **179/200 actions used full depth**, one stopped at layer 6, and twenty at layer 12, skipping **5.38% of blocks**. All twenty layer-12 actions occurred in episode 9, where the model repeatedly chose DOWN into a wall. This is a concrete example of a confident, fast wrong action. The full-depth policy also chose that wrong action; extra computation did not rescue its underlying navigation failure.

The best fixed-depth head on this test was layer 12 at 94/128 optimal actions, compared with 77 at layer 6, 87 at layer 18 and 83 at layer 24. It was not substituted into the episode test after seeing these scores. Selecting a new policy using this result would need fresh evaluation.

### Calibration rejected the candidate

On 128 calibration states, the candidate exited early 42 times. Twelve early actions were not on a shortest path, five were illegal, and six introduced an error relative to full depth. The descriptive state-level upper bounds were:

| Check | Upper bound | Frozen limit | Passed |
| --- | --- | --- | --- |
| Wrong action among early exits | 42.14% | 10% | No |
| New errors across calibration states | 9.04% | 5% | No |
| Illegal action among early exits | 23.42% | 5% | No |

Coverage passed, but all three error checks failed. A guarded policy would keep using full depth. This conclusion concerns the current frozen representation and limited head training, not a claim that every small model or every maze policy must fail.

### Diagnostic times only

The recorded state-pass means were 759.9 ms for the full-depth head and 597.0 ms for the candidate; their medians were 755.1 and 730.3 ms. Concurrent CPU work prevents interpreting that difference as a controlled speedup. BFS averaged 0.047 ms from the structured grid. Its timing excludes ASCII parsing, and its goal completion is the more important comparison here.

The model sees only the instruction and a grid containing its location `A`, goal `G`, walls `#` and open squares `.`. Its four outputs are UP, RIGHT, DOWN and LEFT. No oracle distances, valid-action masks or suggested moves enter the model input. Illegal predictions are retained; they leave the agent in place.

## Frozen experiment design

There are 90 distinct wall layouts. Entire layouts, rather than nearby states from the same layout, are assigned to one partition:

| Partition | Layouts | States |
| --- | --- | --- |
| Train | 32 | 256 |
| Tune | 16 | 128 |
| Calibration | 16 | 128 |
| State test | 16 | 128 |
| Closed-loop episodes | 10 | Up to 20 actions per policy per layout |

Qwen2.5-0.5B-Instruct is frozen. Identical 896-to-4 linear classifiers, with 3,588 learned parameters each, are trained after layers 6, 12, 18 and 24. Standardization uses training data only. Labels are a uniform distribution over **all** legal moves that shorten an optimal route, so a different equally short route is not scored as an error.

Temperature and gate selection use tuning states only. The gate's preset constraints limit loss relative to the full-depth head, errors among early actions and illegal early moves while requiring useful coverage. A separate calibration check can reject it. Candidate traces remain visible even if rejected; a policy respecting rejection would use full depth.

Each layout contributes eight states, which are statistically dependent. The recorded state-level binomial bounds are descriptive checks, not calibrated guarantees over independent mazes or deployment traffic. Twenty actions per episode and ten episode layouts also provide a small, bounded integration test.

## What is actually executed

Every one of the 128 state-test observations receives a fresh full-depth pass and a fresh candidate pass. Block hooks record actual layer indices, and an accepted intermediate head terminates execution immediately. Results must match the corresponding readouts collected during feature extraction. Those full-depth feature passes are not counted as early-exit runtime savings.

The ten episode layouts start at a farthest reachable square. Full-depth and candidate policies receive the same start and goal; each action changes the next observation. Episodes end at the goal or after 20 actions. Failures, wall collisions and loops are retained. The model is not silently replaced with a shortest-path solver.

The explicit cheap control is breadth-first search (BFS), which knows the same grid. It generates training labels and independently chooses its own actions during baseline episodes. Since this environment is deterministic and fully observed, BFS is naturally sufficient. A model has to justify its additional cost rather than merely match this control.

BFS starts from the environment's structured grid; its diagnostic duration does not include parsing ASCII or recognizing a screenshot. Qwen receives the ASCII rendering. This is a comparison of action selection in a known synthetic environment, not a comparison of visual perception systems.

## Verification and interpretation

Environment checks cover boundary wrapping, wall collisions, necessary detours and equally optimal actions. BFS distances were compared with independently implemented all-pairs relaxation. A further independent check recomputed legal actions, all shortest-path choices and distances for every recorded state, and verified all 90 wall layouts are distinct across partitions.

After execution, a separate calculation replayed every transition in all 30 recorded episodes (459 actions), checked legal/optimal flags against independently reconstructed distances, verified all 384 state-control records and actual block traces, and recomputed the result counts. Those checks passed. They verify the recorded mechanism and scoring; they do not turn failed navigation into successful behavior.

All durations from this run are **diagnostic** because another model experiment may run concurrently. They do not establish an isolated latency advantage. Equal poor navigation from two policies would not establish a useful optimization. Goal completion, legal actions and path length on successful episodes are the practical outcomes.

- [Protocol, every maze and every labeled state](../results/maze-actions/protocol.json).
- [Selected gate](../results/maze-actions/selected-policy.json), [state predictions](../results/maze-actions/state-predictions.json), [counted actual execution](../results/maze-actions/runtime-traces.json).
- [Every episode and action](../results/maze-actions/episodes.json), [result summary](../results/maze-actions/result.json), [portable heads](../results/maze-actions/heads.npz).
- [Environment self-checks](../results/maze-actions/environment-checks.json), [implementation](../experiments/maze_actions.py).

Reproduce in the documented local model environment with `python experiments/maze_actions.py --prepare`, then `python experiments/maze_actions.py`. The protocol is fixed before inference and a completed result cannot be silently overwritten. The run uses at most four Torch CPU threads and no generated text tokens.
