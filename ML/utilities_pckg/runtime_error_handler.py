import sys


def runtime_error_handler(str_, add):
    if str_ == "folder_creation_failed":
        sys.exit("Folder creation failed in function " + str(add))

    elif str_ == "shift_not_possible":
        print "It is not possible to execute the require shift operation in function " + str(add) + ". Ignoring..."

    elif str_ == 'no_secret':
        sys.exit("ERROR! The chosen secret " + str(add) + " is not in the secrets set")

    elif str_ == 'not_unique_secr':
        sys.exit("ERROR! The secret " + str(add) + " appears more than once in the channel matrix.")

    elif str_ == 'samples_per_secret':
        sys.exit("ERROR! The required number of samples per secret is None.")

    elif str_ == 'unspecified_option':
        sys.exit("ERROR! Unspecified option in function " + str(add) + ".")

    elif str_ == 'wrong_order_but_right_expected':
        sys.exit("ERROR! In function " + str(add) + " the order wa expected to be correct but is not. Exiting...")


def exception_call(idx, key_val):
    sys.exit(sys.argv[idx + 1].strip() + " is not a valid argument for option " + key_val)
