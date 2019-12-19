from .config import *


# init environment
def init_env():
    # write these imports inside the function to avoid polluting the outer space by using  `import *`
    import os
    import shutil
    import time
    import daemon
    import os.path as osp
    import torch.backends.cudnn as cudnn
    os.environ['CUDA_VISIBLE_DEVICES'] = available_gpus
    cudnn.benchmark = True

    def check_dir(_dir, create=True):
        if not osp.exists(_dir):
            if create:
                os.mkdir(_dir)
            else:
                raise FileNotFoundError("{} not exist".format(_dir))

    check_dir(result_root, create=True)
    check_dir(result_sub_folder, create=True)
    if backup_code:
        print("Working Directory: " + os.path.abspath('.'))
        backup = '{}/{}'.format(result_sub_folder, osp.basename(osp.abspath(".")))
        print('Backup code to {}'.format(backup))
        shutil.copytree(".", backup)

    if daemon_mode:
        # deamon_mode will make the process run in background and no response to signal `HUP`
        log_file = os.path.abspath(os.path.join(result_sub_folder, 'rst.log'))
        print("Daemon mode! stdout and stderr will be redirected to {}".format(log_file))
        outfile = open(log_file, 'w')
        ctx = daemon.DaemonContext(working_directory=os.getcwd(), umask=0o002, stdout=outfile, stderr=outfile)
        ctx.open()
        print("[{}] Daemon mode, pid: {}, GPUs: {}".format(time.asctime(), os.getpid(), available_gpus))
