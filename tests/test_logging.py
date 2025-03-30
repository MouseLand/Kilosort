import pytest
from pathlib import Path
import logging

from kilosort.run_kilosort import setup_logger, close_logger


def test_log(data_directory):
    log_dir = data_directory / 'logging_test'
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir)
    ks_log = logging.getLogger('kilosort')
    ks_log.info('Logging test')

    # Make sure the log is generated in the correct location.
    assert (log_dir / 'kilosort4.log').is_file()
    
    # Need to be able to overwrite the log file if run_kilosort is executed
    # again.
    close_logger()
    setup_logger(log_dir)
    ks_log.info('Logging test 2')
    close_logger()
    with open(log_dir / 'kilosort4.log', mode='r') as f:
        log = f.readlines()
    assert len(log) == 1
    assert log[0].rstrip()[-1] == '2'

    # Should be able to delete the log file after logging is finished.
    (log_dir / 'kilosort4.log').unlink()


def test_log_loop(data_directory):
    # Should be able to run kilosort in a loop and create log files in the
    # correct location each time, without any file errors.
    for i in range(3):
        log_dir = data_directory / 'logging_test' / f'loop_{i}'
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(log_dir)
        ks_log = logging.getLogger('kilosort')
        ks_log.info('Logging test')
        close_logger()
    
    for i in range(3):
        # Make sure each log was generated in the correct location and
        # contains the correct text (i.e. no empty logs or one log containing
        # every iteration).
        log_dir = data_directory / 'logging_test' / f'loop_{i}'
        assert (log_dir / 'kilosort4.log').is_file()
        with open(log_dir / 'kilosort4.log', mode='r') as f:
            log = f.readlines()
        assert len(log) == 1
        (log_dir / 'kilosort4.log').unlink()
