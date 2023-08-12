def select_and_create_subdataset(
    input_dir: str, out_dir: str, selection_method: str = "seq_high_precision"
) -> None:
    file1 = open(input_dir, "r")
    file2 = open(out_dir, "w")

    for line in file1:
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            if (
                int(float(label)) != float(label)
                and selection_method == "seq_high_precision"
            ):
                file2.write(seq + "\t" + label + "\n")
            elif "N" not in seq and selection_method == "complete_seq_only":
                file2.write(seq + "\t" + label + "\n")
