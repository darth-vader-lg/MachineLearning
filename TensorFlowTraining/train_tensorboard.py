# Module: train_tensorboard
#@title #Start the TensorBoard { vertical-output: true }
#@markdown The TensorBoard is run for checking the progress.
#@markdown
#@markdown Warning: an error message will be displayed if no data are yet present.
#@markdown Wait that the train it will be started (loss messages on output) and
#@markdown just click the refresh button.

try:    from base_parameters import BaseParameters
except: pass
import  subprocess
import  sys
import  time
try:    from utilities import *
except: pass

def start_tensorboard(prm: BaseParameters):
    log_dir = os.path.join(prm.model_dir, 'train')
    error = True
    paths = [
        'tensorboard',
        os.path.join(os.path.dirname(sys.executable), 'tensorboard'),
        os.path.join(getattr(sys, '_MEIPASS', sys.executable), 'tensorboard')]
    for tensorboard_path in paths:
        try:
            subprocess.Popen(
                [tensorboard_path, '--port', str(prm.tensorboard_port), '--logdir', log_dir],
                stdout = subprocess.PIPE,
                universal_newlines = True)
            error = False
            break
        except:
            pass
    if (error):
        print('Warning: cannot start tensorboard')
    if (not error and is_jupyter()):
        import tensorboard
        for i in range(5):
            try:
                tensorboard.notebook.display(str(prm.tensorboard_port), 1024)
                break
            except:
                time.sleep(1)

if __name__ == '__main__':
    prm = ('prm' in locals() and isinstance(prm, BaseParameters) and prm) or BaseParameters.default
    start_tensorboard(prm)

#@markdown ---
