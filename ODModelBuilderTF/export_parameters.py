#module export_parameters.py
#@title #Export parameters { form-width: "20%" }
#@markdown Definition of the export parameters. Read the comments in the flags
#@markdown section of the exporter main module
#@markdown https://raw.githubusercontent.com/tensorflow/models/e356598a5b79a768942168b10d9c1acaa923bdb4/research/object_detection/exporter_main_v2.py

import  os

try:    from    base_parameters import BaseParameters
except: pass
try:    from    default_cfg import Cfg
except: pass

class ExportParameters(BaseParameters):
    """ Class holding the model export parameters """
    def __init__(self):
        """ Constructor """
        super().__init__()
        self._input_type = 'image_tensor'
        self._pipeline_config_path = os.path.join(self.model_dir, 'pipeline.config')
        self._trained_checkpoint_dir = self.model_dir
        self._output_directory = Cfg.exported_model
        self._is_path.extend([
            'pipeline_config_path',
            'trained_checkpoint_dir',
            'output_directory'])
    default = None
    @property
    def pipeline_config_path(self): return self._pipeline_config_path
    @pipeline_config_path.setter
    def pipeline_config_path(self, value): self._pipeline_config_path = value
    @property
    def trained_checkpoint_dir(self): return self._trained_checkpoint_dir
    @trained_checkpoint_dir.setter
    def trained_checkpoint_dir(self, value): self._trained_checkpoint_dir = value
    @property
    def output_directory(self): return self._output_directory
    @output_directory.setter
    def output_directory(self, value): self._output_directory = value

ExportParameters.default = ExportParameters.default or ExportParameters()

if __name__ == '__main__':
    prm = ('prm' in locals() and isinstance(prm, ExportParameters) and prm) or ExportParameters.default
    print(prm)
    print('Export parameters configured')

#@markdown ---
