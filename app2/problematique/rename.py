import os
import re


def main():
    # rename all files from a directory
    dir_extension = "street_normal"
    dir = "./images_clean/" + dir_extension
    files = os.listdir(dir)
    print("files", len(files))
    for filename in files:
        class_type = filename.split("_")[0]
        # get the number
        index_raw = re.findall(r"\d+", filename)
        index = int(index_raw[0])

        new_name = dir_extension + "_" + str(index) + ".jpg"

        print(filename, index, class_type, new_name)

        os.rename(dir + "/" + filename, dir + "/" + new_name)


if __name__ == "__main__":
    main()
