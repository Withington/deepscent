import os
import subprocess
import tempfile

import tensorflow as tf
import tensorflow.keras as keras

import nbformat


def notebook_run(path):
    '''Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    '''

    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
          "--ExecutePreprocessor.kernel_name=travis_env",
          "--ExecutePreprocessor.timeout=60",
          "--output", fout.name, path]


        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
                     for output in cell["outputs"]\
                     if output.output_type == "error"]

    return nb, errors


def test_ipynb():
    print('Current working directory is', os.getcwd())
    __, errors = notebook_run('notebooks/deepscent.ipynb')
    assert errors == []


if __name__ == "__main__":
    test_ipynb()
