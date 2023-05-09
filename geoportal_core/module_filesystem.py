import zipfile, os, shutil
from django.conf import settings
import importlib
from pathlib import Path

def get_module_path(module_name, path=settings.MODULE_PATH):
    return os.path.join(path, module_name)

def remove_module_dir(module_name, path=settings.MODULE_PATH):
    if os.path.exists(get_module_path(module_name, path)):
        shutil.rmtree(get_module_path(module_name, path))

def unzip_file(file, module_name, path=settings.MODULE_PATH):

    os.mkdir(get_module_path(module_name, path))

    zfobj = zipfile.ZipFile(file)
    for name in zfobj.namelist():
        zip_base_dir = Path(name).parts[0]
        # there's root dir: archive/root_dir/<zipped_files>
        if zip_base_dir != Path(name).name:
            create_name = name.replace(zip_base_dir, module_name)
        else:
        # all files placed on same level: archive/<zipped_files>
            create_name = os.path.join(module_name, name)
        if name.endswith('/'):
            if not os.path.exists(os.path.join(path, create_name)):
                os.mkdir(create_name)
        else:
            outfile = open(os.path.join(path, create_name), 'wb')
            outfile.write(zfobj.read(name))
            outfile.close()

def try_import(module_name):
    importlib.invalidate_caches()
    config = importlib.import_module(f"{module_name}.module_config", package=None)
    return config.SCHEMA, config.COMMANDS
