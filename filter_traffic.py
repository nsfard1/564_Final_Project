import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python remove_bg_traffic <input.csv> <output.csv>")
        sys.exit(1)

    inputPath = sys.argv[1]
    outputPath = sys.argv[2]

    with open(inputPath, "r") as file:
        lines = file.readlines()
        file.seek(0)
        labels = [line.strip().split(",")[-1] for line in file.readlines()]

    with open(outputPath, 'w') as file:
        for i,line in enumerate(lines):
            if 'Background' not in labels[i]:
                print(line, file=file)

if __name__ == "__main__":
    main()

