#!/usr/bin/env python3
"""Single-worker DeltaStreamer evaluation with the current production profiles."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def make_command(args):
    profile = json.loads((ROOT / 'configs' / f'{args.benchmark}.json').read_text())
    data_root = args.data_root.resolve()
    dataset = (args.dataset or data_root / profile['dataset']).resolve()
    output = args.output.resolve()
    env = dict(os.environ)
    env.update(profile['environment'])
    env.pop('STREAMO_ENABLE_EXACT_UNABLE_DEFAULT', None)
    for key in ['STREAMO_CUDAGRAPH_MODE', 'STREAMO_CUDAGRAPH_CAPTURE_SIZES',
                'STREAMO_ACLGRAPH_DISABLE_INPUT_INPLACE_PASS']:
        env.pop(key, None)
    env['PYTHONPATH'] = str(ROOT / '.runtime')
    env['STREAMO_VIDEO_DIR'] = str(data_root / profile['video_dir'])
    env['STREAMO_WHISPER_DOWNLOAD_ROOT'] = str(ROOT / 'checkpoints' / 'whisper')
    env['STREAMO_SAM2_CKPT'] = str(args.sam2_checkpoint.resolve())
    env['STREAMO_SAM2_CONFIG'] = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    env['STREAMO_MAX_NUM_SEQS'] = str(args.concurrency)
    if args.no_audio:
        env['STREAMO_ASR_BACKEND'] = 'mock'
    if args.no_track:
        env['STREAMO_TRACKER_BACKEND'] = 'mock'
    command = [sys.executable, str(ROOT / 'core/worker.py'),
               '--model', str(args.model.resolve()), '--dataset', str(dataset),
               '--tasks', args.tasks or profile['tasks'], '--tp', str(args.tp),
               '--worker_id', '0', '--num_workers', '1', '--claims_dir', str(output / 'claims'),
               '--claim_batch_size', str(args.concurrency), '--run_tag', 'run',
               '--output_dir', str(output), '--output_prefix', args.benchmark, '--skip_score',
               '--max_concurrent', str(args.concurrency), '--batch_size', str(args.concurrency),
               '--gpu_memory_utilization', str(args.gpu_memory_utilization),
               '--trajectory_dir', str(output / 'trajectories')]
    runtime_args = dict(profile['arguments'])
    if args.no_audio:
        runtime_args['enable_audio_tools'] = False
        runtime_args['auto_asr_every'] = 0
    if args.no_track:
        runtime_args['enable_track_tools'] = False
        runtime_args['auto_track_query_every'] = 0
    for key, value in runtime_args.items():
        if isinstance(value, bool):
            if value:
                command.append('--' + key)
        else:
            command.extend(['--' + key, str(value)])
    return command, env, dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['ovobench', 'streamingbench'], required=True)
    parser.add_argument('--model', type=Path, required=True, help='Local model checkpoint directory.')
    parser.add_argument('--data-root', type=Path, required=True, help='Prepared benchmark directory.')
    parser.add_argument('--dataset', type=Path, help='Override the prepared JSON, e.g. for a subset.')
    parser.add_argument('--output', type=Path, required=True, help='New output directory, or existing directory with --resume.')
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=12)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--tasks', help='Comma-separated tasks; default is the entire benchmark.')
    parser.add_argument('--sam2-checkpoint', type=Path, default=ROOT / 'checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--no-audio', action='store_true', help='Disable ASR; results differ from the full profile.')
    parser.add_argument('--no-track', action='store_true', help='Disable tracking; results differ from the full profile.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without importing models or writing outputs.')
    args = parser.parse_args()
    if args.tp < 1 or args.concurrency < 1 or not 0 < args.gpu_memory_utilization < 1:
        parser.error('tp/concurrency must be positive and gpu-memory-utilization must lie in (0, 1).')
    command, env, dataset = make_command(args)
    if args.dry_run:
        print(shlex.join(command))
        print('Profile:', args.benchmark, '| runtime: .runtime/swift | video directory:', env['STREAMO_VIDEO_DIR'])
        return
    if not (ROOT / '.runtime/runtime.json').is_file():
        parser.error('Run python scripts/setup_runtime.py first.')
    if not (args.model / 'config.json').is_file():
        parser.error('Model config.json not found.')
    if not dataset.is_file():
        parser.error('Prepared dataset JSON not found.')
    if not Path(env['STREAMO_VIDEO_DIR']).is_dir():
        parser.error('Original video directory not found; replay and ASR require it.')
    if not args.no_track and not args.sam2_checkpoint.is_file():
        parser.error('SAM2 checkpoint missing; provide --sam2-checkpoint or explicitly use --no-track.')
    if args.output.exists() and not args.resume:
        parser.error('Output already exists; use --resume or a new output directory.')
    if args.resume and not args.output.is_dir():
        parser.error('--resume requires an existing output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    record = {'benchmark': args.benchmark, 'dataset': str(dataset), 'command': command,
              'environment': {k:v for k,v in env.items() if k.startswith('STREAMO_')}}
    (args.output / ('resume_config.json' if args.resume else 'run_config.json')).write_text(json.dumps(record, indent=2)+'\n')
    subprocess.run(command, env=env, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / f'score_{args.benchmark}.py'),
                    str(args.output.resolve()), '--dataset', str(dataset),
                    '--json-out', str(args.output.resolve() / 'metrics.json')], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
