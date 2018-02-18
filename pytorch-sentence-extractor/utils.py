# encoding: utf-8


def get_models_dir(args):
    return_directory = args.data + "/models/"
    if args.legal:
        return_directory =  args.data + "/models/legal/"
    if args.debug:
        print("Using model directory : " + return_directory)
    return return_directory
