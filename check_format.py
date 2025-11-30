# Open the input and output files
with open('EBD/ebd_US-MI_smp_relSep-2025.txt', 'r') as infile, open('output.txt', 'w') as outfile:
    # Read and write the first 5 lines
    for _ in range(5):
        line = infile.readline()
        if line == '':
            break  # Stop if we reach the end of the input file before 5 lines
        outfile.write(line)
