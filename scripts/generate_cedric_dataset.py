#!/usr/bin/env python3
import json
import random
from pathlib import Path

random.seed(42)
OUT = Path('data/cedric_assistant')
OUT.mkdir(parents=True, exist_ok=True)

samples = []


def add(state_t, state_t1, tag):
    samples.append({'state_t': state_t, 'state_t+1': state_t1, 'tag': tag})


def s_fireplace(user_state, window_state, safety_mode):
    t = [
        ['fireplace', 'state', 'on'],
        ['user', 'state', user_state],
        ['window', 'state', window_state],
        ['safety_mode', 'state', safety_mode],
    ]
    should_off = (user_state == 'away' and window_state == 'open') or (safety_mode == 'strict' and user_state == 'away')
    if should_off:
        n = [
            ['fireplace', 'state', 'off'],
            ['user', 'state', user_state],
            ['window', 'state', window_state],
            ['assistant_action', 'state', 'escalate'],
        ]
    else:
        n = [
            ['fireplace', 'state', 'on'],
            ['user', 'state', user_state],
            ['window', 'state', window_state],
            ['assistant_action', 'state', 'wait'],
        ]
    return t, n


def s_garage(door, user_state, tod):
    t = [
        ['garage_door', 'state', door],
        ['user', 'state', user_state],
        ['time', 'state', tod],
    ]
    should_close = door == 'open' and (user_state == 'away' or tod == 'night')
    if should_close:
        n = [
            ['garage_door', 'state', 'closed'],
            ['user', 'state', user_state],
            ['time', 'state', tod],
            ['assistant_action', 'state', 'ping'],
        ]
    else:
        n = [
            ['garage_door', 'state', door],
            ['user', 'state', user_state],
            ['time', 'state', tod],
            ['assistant_action', 'state', 'wait'],
        ]
    return t, n


def s_reminder(priority, deadline, response, nudge_count):
    t = [
        ['task', 'state', 'pending'],
        ['task', 'priority', priority],
        ['deadline', 'state', deadline],
        ['user', 'state', response],
        ['nudge_count', 'state', nudge_count],
    ]
    if response == 'responded':
        a = 'wait'
    elif priority == 'high' and deadline == 'soon':
        a = 'escalate'
    elif nudge_count in ('zero', 'one'):
        a = 'ping'
    else:
        a = 'wait'
    n = [
        ['task', 'state', 'pending'],
        ['task', 'priority', priority],
        ['deadline', 'state', deadline],
        ['user', 'state', response],
        ['assistant_action', 'state', a],
    ]
    return t, n


def s_focus(energy, meeting, friction, task_state):
    t = [
        ['user', 'state', energy],
        ['calendar', 'state', meeting],
        ['task', 'friction', friction],
        ['task', 'state', task_state],
    ]
    if meeting == 'meeting_now':
        nxt = task_state
        a = 'defer'
    elif energy == 'focused' and friction == 'low_friction':
        nxt = 'done' if task_state == 'started' else 'started'
        a = 'do_next_step'
    else:
        nxt = task_state
        a = 'micro_step'
    n = [
        ['user', 'state', energy],
        ['calendar', 'state', meeting],
        ['task', 'state', nxt],
        ['assistant_action', 'state', a],
    ]
    return t, n


def s_bin_night(day, hour_band, bins_out):
    t = [
        ['trash_day', 'state', day],
        ['time', 'state', hour_band],
        ['bins', 'state', bins_out],
    ]
    # rule: Tue/Wed after 6pm and bins still in -> ping
    should_ping = day in ('tuesday', 'wednesday') and hour_band == 'evening' and bins_out == 'in'
    n = [
        ['trash_day', 'state', day],
        ['time', 'state', hour_band],
        ['bins', 'state', bins_out],
        ['assistant_action', 'state', 'ping' if should_ping else 'wait'],
    ]
    return t, n


# General distributions
for _ in range(260):
    t, n = s_fireplace(
        random.choice(['home', 'away']),
        random.choice(['open', 'closed']),
        random.choice(['strict', 'lenient'])
    )
    add(t, n, 'safety')

for _ in range(220):
    t, n = s_garage(
        random.choice(['open', 'closed']),
        random.choice(['home', 'away']),
        random.choice(['day', 'night'])
    )
    add(t, n, 'garage')

for _ in range(260):
    t, n = s_reminder(
        random.choice(['low', 'medium', 'high']),
        random.choice(['soon', 'later']),
        random.choice(['responded', 'no_response']),
        random.choice(['zero', 'one', 'two'])
    )
    add(t, n, 'reminder')

for _ in range(220):
    t, n = s_focus(
        random.choice(['low_energy', 'focused']),
        random.choice(['meeting_now', 'free_block']),
        random.choice(['high_friction', 'low_friction']),
        random.choice(['pending', 'started'])
    )
    add(t, n, 'focus')

for _ in range(120):
    t, n = s_bin_night(
        random.choice(['monday', 'tuesday', 'wednesday', 'thursday', 'friday']),
        random.choice(['daytime', 'evening']),
        random.choice(['in', 'out'])
    )
    add(t, n, 'bins')

# Build hard context split with deliberate combinations
context = []
for user_state in ['home', 'away']:
    for window_state in ['open', 'closed']:
        for safety_mode in ['strict', 'lenient']:
            for _ in range(8):
                t, n = s_fireplace(user_state, window_state, safety_mode)
                context.append({'state_t': t, 'state_t+1': n})

for door in ['open', 'closed']:
    for user_state in ['home', 'away']:
        for tod in ['day', 'night']:
            for _ in range(6):
                t, n = s_garage(door, user_state, tod)
                context.append({'state_t': t, 'state_t+1': n})

random.shuffle(samples)
train = samples[:900]
test_comp = samples[900:1050]
test_seen = samples[1050:1150]
context = context[:120]

for arr in (train, test_comp, test_seen):
    for ex in arr:
        ex.pop('tag', None)

for name, arr in [
    ('train.jsonl', train),
    ('test_comp.jsonl', test_comp),
    ('test_seen.jsonl', test_seen),
    ('test_context.jsonl', context),
]:
    with (OUT / name).open('w') as f:
        for ex in arr:
            f.write(json.dumps(ex) + '\n')

summary = {
    'train': len(train),
    'test_comp': len(test_comp),
    'test_seen': len(test_seen),
    'test_context': len(context),
}
(OUT / 'README.md').write_text(
    '# Cedric Assistant Dataset\n\n'
    'Domain-specific triples for assistant workflows (safety, garage, reminders, focus, bins).\n\n'
    f'Counts: {summary}\n'
)
print(summary)