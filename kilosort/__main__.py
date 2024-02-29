import argparse 
from kilosort.gui.launch import launcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Kilosort4',
                    description = 'spike-sorting + GUI')
    parser.add_argument('--filename', default=None, type=str, help='filename of binary file to process')
    args = parser.parse_args()
    launcher(args.filename)
