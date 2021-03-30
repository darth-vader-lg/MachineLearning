# Module: mount_google_drive.py
#@title #Mount Google Drive
#@markdown Mounting of the Google Drive (if enabled in the configuration).

import  os
import  sys

try:    from    default_cfg import Cfg
except: pass

def mount_google_drive():
    if (not os.path.exists('/mnt/MyDrive')):
        print('Mounting the GDrive')
        from google.colab import drive
        drive.mount('/mnt')
    else:
        print('GDrive already mounted')

if __name__ == '__main__':
    if (Cfg.data_on_drive and 'google.colab' in sys.modules):
        mount_google_drive()
#@markdown ---
