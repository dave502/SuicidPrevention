import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'SECRET_KEY'

    app_dir = os.path.join(basedir, 'app/')
    source_dir = os.path.join(basedir, 'source/')
    train_test_dir = os.path.join(basedir, 'source/train_test/')
    prediction_dir = os.path.join(basedir, 'source/prediction/')
    dumps_dir = os.path.join(basedir, 'source/dumps/')

