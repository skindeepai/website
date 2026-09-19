"""Bounded maze-action validation: raw ASCII to four Qwen classifier outputs.

Synthetic environments, split by entire wall layout. BFS labels are never part
of model input and never filter its proposed actions. One frozen-backbone run.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np
from scipy.stats import beta
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/maze-actions'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
DEPTHS = (6, 12, 18, 24)
ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT')
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
WIDTH = 4


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transition(maze, position, action):
    row, col = divmod(position, WIDTH)
    dr, dc = DELTAS[action]
    nr, nc = row+dr, col+dc
    if not (0 <= nr < WIDTH and 0 <= nc < WIDTH):
        return position, False
    target = WIDTH*nr+nc
    return (position, False) if target in maze['walls'] else (target, True)


def distances(maze):
    distance = {maze['goal']: 0}
    queue = deque([maze['goal']])
    while queue:
        position = queue.popleft()
        for action in range(4):
            target, legal = transition(maze, position, action)
            if legal and target not in distance:
                distance[target] = distance[position]+1
                queue.append(target)
    return distance


def oracle(maze, position):
    distance = distances(maze)
    legal, optimal = [], []
    for action in range(4):
        target, valid = transition(maze, position, action)
        if valid:
            legal.append(action)
            if distance[target] == distance[position]-1:
                optimal.append(action)
    assert position == maze['goal'] or optimal
    return legal, optimal, distance[position]


def observation(maze, position):
    cells = ['#' if p in maze['walls'] else '.' for p in range(WIDTH*WIDTH)]
    cells[maze['goal']] = 'G'
    cells[position] = 'A'
    return '\n'.join(''.join(cells[i:i+WIDTH]) for i in range(0, WIDTH*WIDTH, WIDTH))


def prompt(maze, position):
    return ('Choose one move that brings A toward G along a shortest path. '
            'A is your location; G is the goal; # is a wall; . is open. '
            'Move one square UP, RIGHT, DOWN, or LEFT. Do not cross walls or the grid edge.\n'
            + observation(maze, position)+'\nMove:')


def self_checks():
    open_grid = {'walls': [], 'goal': 15}
    legal, optimal, distance = oracle(open_grid, 0)
    assert legal == [1, 2] and optimal == [1, 2] and distance == 6
    assert transition(open_grid, 3, 1) == (3, False), 'No horizontal row wrapping'
    blocked = {'walls': [1], 'goal': 3}
    legal, optimal, distance = oracle(blocked, 0)
    assert legal == [2] and optimal == [2] and distance == 5
    assert transition(blocked, 0, 1) == (0, False), 'A wall must not move the agent'
    # Check BFS distances independently against all-pairs relaxation.
    for maze in (open_grid, blocked, {'walls': [5, 6, 9], 'goal': 15}):
        positions = [p for p in range(16) if p not in maze['walls']]
        direct = {(a, b): (0 if a == b else 100) for a in positions for b in positions}
        for a in positions:
            for action in range(4):
                b, valid = transition(maze, a, action)
                if valid:
                    direct[a, b] = 1
        for k in positions:
            for a in positions:
                for b in positions:
                    direct[a, b] = min(direct[a, b], direct[a, k]+direct[k, b])
        assert all(d == direct[p, maze['goal']] for p, d in distances(maze).items())
    return {'passed': True, 'checks': ['Shortest-path tie accepts both moves', 'No boundary wrap',
                                      'Wall collision stays in place', 'Required detour',
                                      'BFS distances agree with independent all-pairs relaxation']}


def prepare():
    rng = random.Random(194319)
    seen = set()
    mazes, states = {}, {split: [] for split in ('train', 'tune', 'calibration', 'test')}
    episode_ids = []
    for split, count in (('train', 32), ('tune', 16), ('calibration', 16), ('test', 16), ('episodes', 10)):
        made = 0
        while made < count:
            walls = tuple(sorted(rng.sample(range(16), 3)))
            if walls in seen:
                continue
            floor = [p for p in range(16) if p not in walls]
            maze = {'id': f'{split}:{made}', 'split': split, 'walls': list(walls), 'goal': rng.choice(floor)}
            distance = distances(maze)
            if len(distance) != len(floor):
                continue
            seen.add(walls)
            available = [p for p in floor if p != maze['goal']]
            rng.shuffle(available)
            if split == 'episodes':
                # A fixed difficult start, chosen solely from the generated maze.
                maze['start'] = max(available, key=lambda p: distance[p])
                episode_ids.append(maze['id'])
            else:
                for position in available[:8]:
                    legal, optimal, shortest = oracle(maze, position)
                    states[split].append({'id': f'{maze["id"]}:{position}', 'maze': maze['id'], 'position': position,
                                          'legal': legal, 'optimal': optimal, 'distance': shortest,
                                          'observation': observation(maze, position)})
            mazes[maze['id']] = maze
            made += 1
    assert len({tuple(m['walls']) for m in mazes.values()}) == len(mazes) == 90
    protocol = {
        'scope': 'Procedural 4x4 static maze, not Pac-Man, natural language traffic, or general action planning.',
        'model': MODEL, 'revision': REVISION, 'seed': 194319, 'head_seed': 97,
        'mazes': mazes, 'states': states, 'episode_ids': episode_ids,
        'split_rule': 'Entire wall layouts disjoint across32/16/16/16 state mazes and10episode mazes;8states per layout.',
        'training': {'depths': list(DEPTHS), 'steps': 200, 'learning_rate': .01, 'weight_decay': .1,
                     'head': '896-to4 linear; training-only mean/std; softmax; uniform target over all optimal next moves'},
        'prompt_template': prompt({'walls': [1, 6, 9], 'goal': 15}, 0),
        'actions': list(ACTIONS), 'temperature_grid': [.5, 1., 1.5, 2., 3., 4., 6., 8.],
        'gate_grid': {'threshold': [.5, .6, .7, .8, .9, .95, .975, .99, 1.01], 'agreement': [False, True], 'minimum': [6, 12]},
        'gate_selection': 'Tuning only: lowest mean depth; <=2pp optimal-action accuracy loss versus full; <=10% wrong early actions; no illegal early actions; >=10% early coverage. Otherwise disabled.',
        'calibration_guard': 'Separate state calibration: individual exact95% upper bounds <=10% wrong/early, <=5% new errors/all, <=5% illegal/early; >=10% coverage. Bounds treat states as samples, though8states share each layout: descriptive, not a maze-cluster guarantee.',
        'episodes': '10new wall layouts, farthest reachable start, same starts for full head and candidate,20moves max. Illegal moves stay in place. No oracle action masking, retries, or resetting failed episodes.',
        'oracle': 'BFS generates labels, checks all equally optimal moves and serves as the explicit cheap control. Oracle outputs never enter the learned observation/readout.',
        'timing': 'All128teststates run full and actualearly paths, rotated order; every executed block recorded. Episode paths run sequentially. Concurrent CPU work makes all times diagnostic, not controlled speedup evidence.',
        'limitations': ['State examples cluster within mazes; generalization unit is the entire held-out maze.',
                        'ASCII navigation can be poor for this small frozen model and limited head training.',
                        'A deterministic fully observed maze is naturally solved by BFS; an LLM must justify additional cost.',
                        'One model run; no threshold adjustment after calibration or test results.'],
        'script_sha256': digest(Path(__file__))
    }
    existing = OUT/'protocol.json'
    if existing.exists():
        assert json.loads(existing.read_text()) == protocol, 'Protocol changed; do not overwrite an existing experiment.'
    else:
        write('protocol.json', protocol)
    return protocol


def upper(k, n):
    return 1. if not n or k == n else float(beta.ppf(.95, k+1, n-k))


def summarize(rows, predictions, depths, full):
    correct = [p in r['optimal'] for r, p in zip(rows, predictions)]
    early = [i for i, d in enumerate(depths) if d < 24]
    wrong = sum(not correct[i] for i in early)
    illegal = sum(predictions[i] not in rows[i]['legal'] for i in early)
    harms = sum(f in r['optimal'] and not c for r, f, c in zip(rows, full, correct))
    return {'n': len(rows), 'optimal_correct': sum(correct), 'legal_actions': sum(p in r['legal'] for r, p in zip(rows, predictions)),
            'full_optimal_correct': sum(p in r['optimal'] for r, p in zip(rows, full)),
            'early_count': len(early), 'early_wrong': wrong, 'early_illegal': illegal, 'added_errors': harms,
            'mean_depth': statistics.mean(depths), 'exit_counts': dict(Counter(depths)),
            'blocks_skipped': 1-statistics.mean(depths)/24,
            'wrong_early_upper95': upper(wrong, len(early)), 'illegal_early_upper95': upper(illegal, len(early)),
            'added_error_upper95': upper(harms, len(rows))}


def choose(probs, threshold, agreement, minimum):
    n = len(probs[24])
    predictions, depths = probs[24].argmax(1).tolist(), [24]*n
    previous = None
    for depth in DEPTHS[:-1]:
        confidence, candidate = probs[depth].max(1)
        for i in range(n):
            if depths[i] == 24 and depth >= minimum and float(confidence[i]) >= threshold and (not agreement or (previous is not None and int(candidate[i]) == previous[i])):
                predictions[i], depths[i] = int(candidate[i]), depth
        previous = candidate.tolist()
    return predictions, depths


class StopMove(Exception):
    def __init__(self, prediction, depth):
        self.prediction, self.depth = prediction, depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(min(4, max(1, (os.cpu_count() or 2)//2)))
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    checks = self_checks()
    protocol = prepare()
    write('environment-checks.json', checks)
    if args.prepare:
        print('Recorded maze protocol and passed environment checks.', flush=True)
        return
    assert not (OUT/'result.json').exists(), 'Completed experiment exists; no silent replacement.'
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                                               torch_dtype=torch.float32, attn_implementation='eager').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert len(model.model.layers) == 24 and model.config.hidden_size == 896
    mazes, splits = protocol['mazes'], protocol['states']

    def inputs(rows):
        messages = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt(mazes[r['maze']], r['position'])}],
                                                  tokenize=False, add_generation_prompt=True) for r in rows]
        return tokenizer(messages, return_tensors='pt', padding=True)

    features = {}
    with torch.inference_mode():
        for split, rows in splits.items():
            collected = {d: [] for d in DEPTHS}
            for start in range(0, len(rows), 4):
                handles = [model.model.layers[d-1].register_forward_hook(lambda module, args, output, d=d: collected[d].append(output[0][:, -1, :].clone())) for d in DEPTHS[:-1]]
                try:
                    result = model.model(**inputs(rows[start:start+4]), use_cache=False)
                finally:
                    for handle in handles:
                        handle.remove()
                collected[24].append(result.last_hidden_state[:, -1, :].clone())
                if start % 64 == 0:
                    print(f'Features {split} {min(start+4, len(rows))}/{len(rows)}', flush=True)
            features[split] = {d: torch.cat(v).clone() for d, v in collected.items()}
    # Training needs ordinary tensors outside inference mode.
    features = {s: {d: x.clone() for d, x in layers.items()} for s, layers in features.items()}
    targets = {}
    for split, rows in splits.items():
        targets[split] = torch.tensor([[1/len(row['optimal']) if action in row['optimal'] else 0 for action in range(4)] for row in rows])
    heads, probabilities, portable = {}, {s: {} for s in splits}, {}
    torch.manual_seed(97)
    for depth in DEPTHS:
        x = features['train'][depth]
        mean, std = x.mean(0), x.std(0).clamp_min(.05)
        head = torch.nn.Linear(896, 4)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
        for _ in range(200):
            optimizer.zero_grad()
            loss = -(targets['train']*torch.log_softmax(head((x-mean)/std), 1)).sum(1).mean()
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            tune = head((features['tune'][depth]-mean)/std)
            temperature = min(protocol['temperature_grid'], key=lambda t: float(-(targets['tune']*torch.log_softmax(tune/t, 1)).sum(1).mean()))
            for split in splits:
                probabilities[split][depth] = torch.softmax(head((features[split][depth]-mean)/std)/temperature, 1)
        heads[depth] = (head, mean, std, temperature)
        for name, value in [('weight', head.weight), ('bias', head.bias), ('mean', mean), ('std', std)]:
            portable[f'{depth}_{name}'] = value.detach().numpy()
        portable[f'{depth}_temperature'] = np.array(temperature)
    np.savez_compressed(OUT/'heads.npz', **portable)
    candidates = []
    tune_full = probabilities['tune'][24].argmax(1).tolist()
    for minimum in (6, 12):
        for agreement in (False, True):
            for threshold in protocol['gate_grid']['threshold']:
                predictions, depths = choose(probabilities['tune'], threshold, agreement, minimum)
                metrics = summarize(splits['tune'], predictions, depths, tune_full)
                if (metrics['early_count']/metrics['n'] >= .1 and metrics['early_illegal'] == 0
                        and (metrics['optimal_correct']-metrics['full_optimal_correct'])/metrics['n'] >= -.02
                        and metrics['early_wrong']/max(1, metrics['early_count']) <= .1):
                    candidates.append({'threshold': threshold, 'agreement': agreement, 'minimum': minimum, 'tune': metrics})
    chosen = min(candidates, key=lambda x: (x['tune']['mean_depth'], -x['threshold'])) if candidates else {
        'threshold': 1.01, 'agreement': False, 'minimum': 6, 'reason': 'No tuning candidate met the frozen constraints'}
    write('selected-policy.json', chosen)
    result = {'protocol_sha256': digest(OUT/'protocol.json'), 'policy': chosen, 'candidate_count': len(candidates), 'fixed_heads': {}, 'states': {}}
    traces = []
    for split in ('calibration', 'test'):
        full = probabilities[split][24].argmax(1).tolist()
        predictions, depths = choose(probabilities[split], chosen['threshold'], chosen['agreement'], chosen['minimum'])
        result['states'][split] = summarize(splits[split], predictions, depths, full)
        for i, row in enumerate(splits[split]):
            traces.append({'split': split, **row, 'full_prediction': full[i], 'prediction': predictions[i], 'depth': depths[i],
                           'heads': {d: {'prediction': int(probabilities[split][d][i].argmax()), 'confidence': float(probabilities[split][d][i].max())} for d in DEPTHS}})
    for depth in DEPTHS:
        predictions = probabilities['test'][depth].argmax(1).tolist()
        result['fixed_heads'][depth] = summarize(splits['test'], predictions, [depth]*len(predictions), probabilities['test'][24].argmax(1).tolist())
    cal = result['states']['calibration']
    result['guard_checks'] = {'absolute_early_error': cal['wrong_early_upper95'] <= .1, 'added_error': cal['added_error_upper95'] <= .05,
                              'illegal_early': cal['illegal_early_upper95'] <= .05, 'coverage': cal['early_count']/cal['n'] >= .1}
    result['guard_passed'] = all(result['guard_checks'].values())
    write('state-predictions.json', traces)

    def score(hidden, depth):
        head, mean, std, temperature = heads[depth]
        probability = torch.softmax(head((hidden-mean)/std)/temperature, 1)[0]
        return int(probability.argmax()), float(probability.max())

    def execute(row, adaptive):
        start = time.perf_counter()
        batch = inputs([row])
        visited, examined, handles = [], [], []

        def inspect_block(depth, module, arguments, output):
            visited.append(depth)
            if adaptive and depth in DEPTHS[:-1]:
                prediction, confidence = score(output[0][:, -1, :], depth)
                agrees = not chosen['agreement'] or bool(examined and examined[-1]['prediction'] == prediction)
                examined.append({'depth': depth, 'prediction': prediction, 'confidence': confidence})
                if depth >= chosen['minimum'] and confidence >= chosen['threshold'] and agrees:
                    raise StopMove(prediction, depth)

        for index, block in enumerate(model.model.layers, start=1):
            handles.append(block.register_forward_hook(partial(inspect_block, index)))
        try:
            output = model.model(**batch, use_cache=False)
            prediction, _ = score(output.last_hidden_state[:, -1, :], 24)
            depth = 24
        except StopMove as stop:
            prediction, depth = stop.prediction, stop.depth
        finally:
            for handle in handles:
                handle.remove()
        assert visited == list(range(1, depth+1))
        return {'prediction': prediction, 'depth': depth, 'executed_layers': visited, 'head_checks': examined,
                'diagnostic_input_to_action_ms': 1000*(time.perf_counter()-start)}

    runtime, episodes = [], []
    with torch.inference_mode():
        execute(splits['test'][0], False)
        execute(splits['test'][0], True)
        reference = {row['id']: row for row in traces if row['split'] == 'test'}
        for i, row in enumerate(splits['test']):
            for adaptive in ((False, True) if i % 2 == 0 else (True, False)):
                actual = execute(row, adaptive)
                expected = reference[row['id']]
                assert actual['prediction'] == expected['prediction' if adaptive else 'full_prediction']
                assert actual['depth'] == (expected['depth'] if adaptive else 24)
                runtime.append({'id': row['id'], 'path': 'candidate' if adaptive else 'full', **actual})
            start = time.perf_counter()
            _, optimal, _ = oracle(mazes[row['maze']], row['position'])
            elapsed = 1000*(time.perf_counter()-start)
            runtime.append({'id': row['id'], 'path': 'bfs', 'prediction': optimal[0], 'depth': 0, 'executed_layers': [], 'diagnostic_input_to_action_ms': elapsed})
            if i % 32 == 0:
                print(f'Actual state execution {i+1}/128', flush=True)
        write('runtime-traces.json', runtime)
        for maze_id in protocol['episode_ids']:
            maze = mazes[maze_id]
            for path in ('full', 'candidate', 'bfs'):
                position = maze['start']
                shortest = distances(maze)[position]
                steps = []
                for step in range(20):
                    if position == maze['goal']:
                        break
                    start = time.perf_counter()
                    if path == 'bfs':
                        _, optimal, _ = oracle(maze, position)
                        actual = {'prediction': optimal[0], 'depth': 0, 'executed_layers': [], 'diagnostic_input_to_action_ms': 1000*(time.perf_counter()-start)}
                    else:
                        actual = execute({'maze': maze_id, 'position': position}, path == 'candidate')
                    legal, optimal, distance = oracle(maze, position)
                    target, valid = transition(maze, position, actual['prediction'])
                    steps.append({'step': step, 'position': position, 'next_position': target, 'legal': valid,
                                  'optimal': actual['prediction'] in optimal, 'distance_before': distance, **actual})
                    position = target
                reached = position == maze['goal']
                episodes.append({'maze': maze_id, 'path': path, 'start': maze['start'], 'goal': maze['goal'], 'shortest_path_length': shortest,
                                 'goal_reached': reached, 'actions': len(steps), 'legal_actions': sum(s['legal'] for s in steps),
                                 'path_length_ratio_if_success': len(steps)/shortest if reached else None, 'steps': steps})
                print(f'Episode {maze_id} {path}: reached={reached}, actions={len(steps)}', flush=True)
            write('episodes.json', episodes)
    result['episode_summary'] = {}
    result['diagnostic_state_timing'] = {}
    for path in ('full', 'candidate', 'bfs'):
        group = [row for row in episodes if row['path'] == path]
        result['episode_summary'][path] = {'episodes': len(group), 'goals_reached': sum(row['goal_reached'] for row in group),
                                          'actions': sum(row['actions'] for row in group), 'legal_actions': sum(row['legal_actions'] for row in group),
                                          'successful_path_ratios': [row['path_length_ratio_if_success'] for row in group if row['goal_reached']]}
        samples = [row['diagnostic_input_to_action_ms'] for row in runtime if row['path'] == path]
        result['diagnostic_state_timing'][path] = {'n': len(samples), 'mean_ms': statistics.mean(samples), 'p50_ms': statistics.median(samples)}
    result['environment'] = {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,
                             'threads': torch.get_num_threads(), 'model': MODEL, 'revision': REVISION, 'completed_utc': datetime.now(timezone.utc).isoformat()}
    result['limitations'] = protocol['limitations'] + ['All timings diagnostic because concurrent CPU workloads were allowed.',
        'State-level confidence bounds ignore within-maze dependence and must not be interpreted as deployment guarantees.',
        'If calibration fails, candidate traces are exploratory; accepted guarded policy would use full depth.']
    write('result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
