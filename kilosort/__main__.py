import argparse 
from kilosort.gui.launch import launcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Kilosort4',
        description='Spike sorting GUI'
        )
    parser.add_argument(
        '--filename', default=None, type=str,
        help='filename of binary file to process'
        )
    parser.add_argument(
        '--reset', action='store_true',
        help='Clears all cached settings before opening the GUI.'
    )

    args = parser.parse_args()
    launcher(args.filename, args.reset)
