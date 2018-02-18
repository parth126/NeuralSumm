# encoding: utf-8


def get_models_dir(args):
    return_directory = args.data + "/models/"
    if args.legal:
        return_directory =  args.data + "/models/legal/"
    if args.debug:
        print("Using model directory : " + return_directory)
    return return_directory

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def return_color(int_nu):
        int_n = int_nu[0]
        if int_n > 0.05:
            return bcolors.OKGREEN
        elif int_n < 0.01:
            return bcolors.FAIL
        elif int_n > 0.03:
            return bcolors.OKBLUE
        else:
            return bcolors.WARNING
