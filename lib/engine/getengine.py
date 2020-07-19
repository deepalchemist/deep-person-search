import shutil
import inspect
import importlib
from lib.engine.twostage import TwoStage

__abbr_to_fullname = {
    'bsl': 'bsl',
    'oim': 'oim',
    'rtn': 'retina',
    'ssd': 'ssd',
    'nae': 'nae'
}


def find_engine_using_name(engine_name):
    engine_name = __abbr_to_fullname[engine_name]
    # Given the option --model [modelname],
    # the file "engine/enginename.py" will be imported.
    engine_filename = 'lib.engine.' + engine_name
    enginelib = importlib.import_module(engine_filename)

    # In the file, the class called ModelNameEngine() will
    # be instantiated. It has to be a subclass of BaseEngine,
    # and it is case-insensitive.
    engine = None
    target_model_name = engine_name.replace('_', '')
    for name, cls in enginelib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, TwoStage):
            engine = cls

    if engine is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase."
              % (engine_filename, target_model_name))
        exit(0)

    return engine


def get_config_setter(engine_name):
    engine_class = find_engine_using_name(engine_name)
    return engine_class.modify_commandline_options


def create_engine(cfg):
    engine = find_engine_using_name(cfg.backbone)
    instance = engine(cfg)
    shutil.copy(inspect.getfile(engine), cfg.expr_dir)
    print("Engine [%s] was created" % (instance.name()))
    return instance
