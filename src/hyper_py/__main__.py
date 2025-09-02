# Allows: python -m hyper_py [optional /path/to/config.yaml]
# This delegates to the same CLI logic used by the console_script entry point.

from .run_hyper import main

if __name__ == "__main__":
    main()
