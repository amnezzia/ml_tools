#
#
#
#
#
################################################

import os
import time

import json
import pickle





FILE_CREATION_LOG_FILE_NAME = '__files_log.log'
SCRIPT_RUN_LOG_FILE_NAME = '__script_run.log'

FILE_CREATION_LOG_TEMPLATE = '''
time: {}
script: {}
wrote file: {}
shape: {}
'''

def log_output(fname, _options, output_shape=''):

    if not os.path.exists(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', ''))):
        #os.chdir(_options.get('root_dir', ''))
        os.makedirs(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', '')), exist_ok=True)

    with open(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', ''), FILE_CREATION_LOG_FILE_NAME), 'a') as f:
        f.write(FILE_CREATION_LOG_TEMPLATE.format(time.strftime('%Y-%m-%d %H:%M:%S'),
                                                  _options['script_file'],
                                                  fname,
                                                  output_shape))


def log_script_run(_options):

    if not os.path.exists(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', ''))):
        os.makedirs(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', '')), exist_ok=True)

    with open(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', ''), SCRIPT_RUN_LOG_FILE_NAME), 'a') as f:
        f.write('\n\n')
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
        f.write('\n')
        json.dump(obj=_options, fp=f, indent=4, sort_keys=True)


def save_df(df, fname, _options, archive=False):
    if not os.path.exists(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', ''))):
        os.makedirs(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', '')), exist_ok=True)

    fpath = os.path.join(_options['root_dir'], _options['output_dir'], fname)
    _options['output_files'].append(fpath)

    output_shapes = _options.get('output_shapes', [])
    output_shapes.append(df.shape)
    _options['output_shapes'] = output_shapes

    if archive:

        df.to_csv(fpath + '.zip', index=False, compression='bz2')
        log_output(fname + '.zip', _options, output_shape='{}, {}'.format(*df.shape))
    else:
        df.to_csv(fpath, index=False)
        log_output(fname, _options, output_shape='{}, {}'.format(*df.shape))

    return _options


def save_model(model, fname, _options):
    '''
    for gensim models
    :param model:
    :param fname:
    :param _options:
    :return:
    '''
    if not os.path.exists(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', ''))):
        os.makedirs(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', '')), exist_ok=True)

    fpath = os.path.join(_options['root_dir'], _options['output_dir'], fname)
    _options['output_files'].append(fpath)

    model.save(fpath)

    log_output(fname, _options)

    return _options


def save_corpus(corp, fname, _options):
    '''
    for gensim corpora
    :param corp:
    :param fname:
    :param _options:
    :return:
    '''
    if not os.path.exists(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', ''))):
        os.makedirs(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', '')), exist_ok=True)

    fpath = os.path.join(_options['root_dir'], _options['output_dir'], fname)
    _options['output_files'].append(fpath)

    g.corpora.MmCorpus.serialize(fpath, corp)

    log_output(fname, _options)

    return _options


def save_object(obj, fname, _options):
    '''
    for pickling any python object
    :param obj:
    :param fname:
    :param _options:
    :return:
    '''
    if not os.path.exists(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', ''))):
        os.makedirs(os.path.join(_options.get('root_dir', ''), _options.get('output_dir', '')), exist_ok=True)

    fpath = os.path.join(_options['root_dir'], _options['output_dir'], fname)
    _options['output_files'].append(fpath)

    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)

    log_output(fname, _options, output_shape='{}, {}'.format(*getattr(obj, 'shape', ['na', 'na', ])))

    return _options


def load_object(fpath):
    '''
    unpickle python object
    :param fpath:
    :return:
    '''
    if not os.path.exists(fpath):
        print(' - - - - - did not find {}'.format(fpath))
        return

    with open(fpath, 'rb') as f:
        obj = pickle.load(f)

    return obj


def get_batches(some_list, n_batches):
    '''
    slice some array-like object into n_batches, this is mainly for multiprocessing
    :param some_list:
    :param n_batches:
    :return:
    '''
    l = len(some_list)
    bs = int(l / n_batches) + 1
    for i in range(n_batches):
        yield some_list[i * bs : (i + 1) * bs]

