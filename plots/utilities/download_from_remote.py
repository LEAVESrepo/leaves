import os, sys


def createFolder(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            sys.exit("\nCreation of the directory %s failed" % path)
        else:
            print("\nSuccessfully created the directory %s " % path)
    else:
        print("\nDirectory %s already existing." % path)


def download_directory_from_server(remote_obj,
                                   local_folder=os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/tmp_res',
                                   user='mromanel',
                                   server='selene.saclay.inria.fr'):
    createFolder(path=local_folder)

    cmd = "scp -r " + user + '@' + server + ':' + remote_obj + ' ' + local_folder
    try:
        os.system(cmd)
        return [True, local_folder]
    except os.system(cmd) != 0:
        print("Command failed execution")
        return [False, None]
