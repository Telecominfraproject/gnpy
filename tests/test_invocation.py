# -*- coding: utf-8 -*-

from pathlib import Path
import os
import pytest
import subprocess
from gnpy.tools.cli_examples import transmission_main_example, path_requests_run

SRC_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("output, handler, args", (
    ('transmission_main_example', transmission_main_example, []),
    ('path_requests_run', path_requests_run, []),
    ('transmission_main_example__raman', transmission_main_example,
     ['examples/raman_edfa_example_network.json', '--sim', 'examples/sim_params.json', '--show-channels',]),
))
def test_example_invocation(capfdbinary, output, handler, args):
    '''Make sure that our examples produce useful output'''
    os.chdir(SRC_ROOT)
    expected = open(SRC_ROOT / 'tests' / 'invocation' / output, mode='rb').read()
    handler(args)
    captured = capfdbinary.readouterr()
    assert captured.out == expected
    assert captured.err == b''

@pytest.mark.parametrize('program', ('gnpy-transmission-example', 'gnpy-path-request'))
def test_run_wrapper(program):
    '''Ensure that our wrappers really, really work'''
    proc = subprocess.run((program, '--help'), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True, universal_newlines=True)
    assert proc.stderr == ''
    assert 'https://github.com/telecominfraproject/oopt-gnpy' in proc.stdout.lower()
    assert 'https://gnpy.readthedocs.io/' in proc.stdout.lower()
