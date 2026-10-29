"""Build a small Plotly Cloud upload folder without training data or environments.

Run: python scripts/prepare_cloud.py
Optional: python scripts/prepare_cloud.py --model chess-npc.npz
"""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,help='Include this trained NPC artifact as chess-npc.npz.')
    args = parser.parse_args()
    if args.model and not args.model.is_file():
        parser.error(f'Model not found: {args.model}')
    root = Path(__file__).resolve().parent.parent
    destination = root / 'deploy' / 'plotly-cloud'
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('app.py','main.py','requirements.txt'):
        shutil.copy2(root/name,destination/name)
    for directory, pattern in (('plotly_chess','*.py'),('assets','*')):
        target = destination/directory
        target.mkdir(exist_ok=True)
        for source in (root/directory).glob(pattern):
            if source.is_file():
                shutil.copy2(source,target/source.name)
    for cache in destination.rglob('__pycache__'):
        if cache.is_dir():
            shutil.rmtree(cache)
    model_target = destination/'chess-npc.npz'
    if args.model:
        shutil.copy2(args.model,model_target)
    elif model_target.exists():
        # Remove only the previously generated model inside the known upload directory.
        model_target.unlink()
    print(f'Upload folder: {destination}')
    print('Upload its contents, preserving the assets/ and plotly_chess/ folders.')
    print('Choose app.py as the main file and Python 3.11.')
    print('Persistent filesystem: automatically detects "Games" mount (or GAMES_DIR).')
    if args.model:
        print('Set the Cloud environment variable PLOTLY_CHESS_MODEL=chess-npc.npz')
    else:
        print('No model included: the app will use material search.')


if __name__ == '__main__':
    main()
