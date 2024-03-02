import argparse
import os

def main():
    parse = argparse.ArgumentParser()

    parse.add_argument("--input_path", type=str, required=True)
    parse.add_argument("--output_path", type=str, required=True)
    parse.add_argument("--model_name", type=str, required=True)

    args = parse.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name

    fout = open(output_path, 'a')
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '' and line.split()[0] == model_name:
                print(args.input_path, ':', line.split()[3], file=fout)
                print(args.input_path, ":", line.split()[3])

    fout.close()

if __name__ == "__main__":
    main()
