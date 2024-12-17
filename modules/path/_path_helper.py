from pathlib import Path

def get_project_root():
    """
    Get the root directory of the project.

    This function determines the project's root directory based on the file's location
    when running in a script or the current working directory when running in a Jupyter Notebook.

    Returns
    -------
    pathlib.Path
        The project's root directory as a Path object.

    Notes
    -----
    - If executed in a script, the root is determined relative to the file's location using `__file__`.
    - If executed in a Jupyter Notebook, it falls back to the current working directory using `Path.cwd()`.
    """
    try:
        # When running in a script
        return Path(__file__).resolve().parent.parent
    except NameError:
        # Fallback for Jupyter Notebook
        return Path.cwd()
