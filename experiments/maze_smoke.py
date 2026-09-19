"""Exploratory structured-grid learned policy, with explicit legal-action control."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import copy
import hashlib
import json
from collections import deque
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/maze-smoke'
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def transition(maze, pos, action):
    r, c = divmod(pos, 4)
    dr, dc = DELTAS[action]
    nr, nc = r+dr, c+dc
    if not (0 <= nr < 4 and 0 <= nc < 4) or nr*4+nc in maze['walls']:
        return pos, False
    return nr*4+nc, True


def distances(maze):
    found = {maze['goal']: 0}
    queue = deque([maze['goal']])
    while queue:
        pos = queue.popleft()
        for action in range(4):
            nxt, legal = transition(maze, pos, action)
            if legal and nxt not in found:
                found[nxt] = found[pos]+1
                queue.append(nxt)
    return found


def optimal(maze, pos):
    ds = distances(maze)
    return [a for a in range(4) if transition(maze, pos, a)[1] and ds[transition(maze, pos, a)[0]] == ds[pos]-1]


def encode(maze, pos):
    grid = torch.zeros(3, 16)
    grid[0, maze['walls']] = 1
    grid[1, pos] = 1
    grid[2, maze['goal']] = 1
    return grid.flatten()


def predict(model, maze, pos, masked):
    # Deliberately no search, distance, optimal-move label or path history.
    logits = model(encode(maze, pos).unsqueeze(0))[0]
    if masked:
        logits = logits.clone()
        for action in range(4):
            if not transition(maze, pos, action)[1]:
                logits[action] = -torch.inf
    return int(logits.argmax())


def rollout(model, maze, masked=False, bfs=False):
    pos = maze['start']
    steps = []
    for i in range(20):
        if pos == maze['goal']:
            break
        action = optimal(maze, pos)[0] if bfs else predict(model, maze, pos, masked)
        nxt, legal = transition(maze, pos, action)
        steps.append({'step': i, 'position': pos, 'prediction': action, 'next_position': nxt,
                      'legal': legal, 'optimal': action in optimal(maze, pos)})
        pos = nxt
    return {'maze': maze['id'], 'start': maze['start'], 'goal': maze['goal'], 'goal_reached': pos == maze['goal'],
            'actions': len(steps), 'legal_actions': sum(r['legal'] for r in steps),
            'shortest_path_length': distances(maze)[maze['start']], 'steps': steps}


def states_report(model, rows, mazes, masked):
    records = []
    for r in rows:
        pred = predict(model, mazes[r['maze']], r['position'], masked)
        records.append({'id': r['id'], 'prediction': pred, 'optimal': pred in r['optimal'],
                        'legal': pred in r['legal']})
    return {'n': len(records), 'optimal_correct': sum(r['optimal'] for r in records),
            'legal_actions': sum(r['legal'] for r in records)}, records


def episodes_report(records):
    return {'n': len(records), 'goals_reached': sum(r['goal_reached'] for r in records),
            'actions': sum(r['actions'] for r in records), 'legal_actions': sum(r['legal_actions'] for r in records)}


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    parent_path = ROOT/'results/maze-actions/protocol.json'
    parent = json.loads(parent_path.read_text())
    mazes = parent['mazes']
    assert len({tuple(m['walls']) for m in mazes.values()}) == len(mazes) == 90
    protocol = {'scope': 'Exploratory follow-up on previously inspected procedural 4x4 mazes, not real-world navigation.',
                'parent_sha256': hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'train': 'Only original32train wall layouts. Compare original256labeledstates with all4992reachable agent/goal pairs from these same layouts.',
                'architecture': '48onehot wall/agent/goal inputs ->128GELU->128GELU->4logits. AdamW lr.002 weight_decay.01; seed913;600steps, random minibatch256; soft targets uniformly cover all shortest-path ties.',
                'selection': 'Checkpoints100,300,600 ranked by unmasked goal success on16tune layouts with farthest starts, then tune128state optimal accuracy; ties earliest checkpoint. No evaluation selection.',
                'evaluation': 'Report both unmasked and explicitly legal-masked selected model on old128calibrationstates,128teststates and10reusedepisodes;20moves. Same weights for masked and unmasked.',
                'mask': 'Uses visible walls and grid boundaries only; forbids illegal actions. No BFS/distances/path history available to learned prediction. Mask does not prevent legal loops.',
                'comparison': 'Historical frozenQwen rawASCII results retained as context, not matched architecture/representation/training. Qwen mask untested because saved peraction logits unavailable.',
                'timing': 'No model speed claims; two CPU threads. Goal outcomes and legality only.'}
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        write('protocol.json', protocol)
    tune_mazes = []
    for m in mazes.values():
        if m['split'] == 'tune':
            ds = distances(m)
            tune_mazes.append(dict(m, start=max(ds, key=ds.get)))
    histories, results, episode_records, state_records = {}, {}, {}, {}
    for variant in ['original_states', 'all_train_goal_pairs']:
        torch.manual_seed(913)
        training = []
        if variant == 'original_states':
            for row in parent['states']['train']:
                training.append((encode(mazes[row['maze']], row['position']), row['optimal']))
        else:
            for m in mazes.values():
                if m['split'] != 'train':
                    continue
                floor = [p for p in range(16) if p not in m['walls']]
                for goal in floor:
                    variant_maze = dict(m, goal=goal)
                    for pos in floor:
                        if pos != goal:
                            training.append((encode(variant_maze, pos), optimal(variant_maze, pos)))
        x = torch.stack([r[0] for r in training])
        target = torch.zeros(len(training), 4)
        for i, (_, actions) in enumerate(training):
            target[i, actions] = 1/len(actions)
        model = torch.nn.Sequential(torch.nn.Linear(48, 128), torch.nn.GELU(), torch.nn.Linear(128, 128),
                                    torch.nn.GELU(), torch.nn.Linear(128, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
        history, best = [], None
        for step in range(1, 601):
            index = torch.randint(len(x), (256,))
            optimizer.zero_grad()
            loss = -(target[index]*model(x[index]).log_softmax(1)).sum(1).mean()
            loss.backward()
            optimizer.step()
            if step in [100, 300, 600]:
                with torch.inference_mode():
                    state, _ = states_report(model, parent['states']['tune'], mazes, False)
                    eps = episodes_report([rollout(model, m) for m in tune_mazes])
                rank = (eps['goals_reached'], state['optimal_correct'])
                history.append({'step': step, 'loss': float(loss.detach()), 'tune_states': state, 'tune_episodes': eps})
                if best is None or rank > best[0]:
                    best = (rank, step, copy.deepcopy(model.state_dict()))
        model.load_state_dict(best[2])
        model.eval()
        np.savez_compressed(OUT/f'{variant}.npz', **{k: v.numpy() for k, v in model.state_dict().items()})
        histories[variant] = {'training_states': len(training), 'selected_step': best[1], 'checkpoints': history}
        for masked in [False, True]:
            method = variant + ('_masked' if masked else '_unmasked')
            with torch.inference_mode():
                metrics = {}
                state_records[method] = {}
                for split in ['calibration', 'test']:
                    metrics[split], state_records[method][split] = states_report(model, parent['states'][split], mazes, masked)
                episode_records[method] = [rollout(model, mazes[mid], masked) for mid in parent['episode_ids']]
            metrics['episodes'] = episodes_report(episode_records[method])
            results[method] = metrics
    episode_records['bfs'] = [rollout(None, mazes[mid], bfs=True) for mid in parent['episode_ids']]
    results['bfs'] = {'episodes': episodes_report(episode_records['bfs']),
                      'test': {'n': 128, 'optimal_correct': 128, 'legal_actions': 128}}
    previous = json.loads((ROOT/'results/maze-actions/episodes.json').read_text())
    for path in ['full', 'candidate']:
        results[f'historical_qwen_{path}'] = {'episodes': episodes_report([r for r in previous if r['path'] == path])}
    write('history.json', histories)
    write('result.json', results)
    write('episodes.json', episode_records)
    write('states.json', state_records)
    print(json.dumps({'training': histories, 'results': results}), flush=True)


if __name__ == '__main__':
    main()
