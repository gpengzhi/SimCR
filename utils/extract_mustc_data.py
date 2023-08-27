import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, type=str)
    parser.add_argument("--dir-in", required=True, type=str)
    parser.add_argument("--dir-out", required=True, type=str)
    parser.add_argument("--file", required=True, type=str)
    args = parser.parse_args()

    tsv_lines = open("{}/{}.tsv".format(args.dir_in, args.file)).readlines()
    src = open("{}/{}.en".format(args.dir_out, args.file), "w")
    tgt = open("{}/{}.{}".format(args.dir_out, args.file, args.lang), "w")

    for i in range(1, len(tsv_lines)):
        _, _, _, src_sentence, tgt_sentence, _ = tsv_lines[i].rstrip('\n').split('\t')
        src.write(src_sentence + '\n')
        tgt.write(tgt_sentence + '\n')
 

if __name__ == "__main__":
    main()
