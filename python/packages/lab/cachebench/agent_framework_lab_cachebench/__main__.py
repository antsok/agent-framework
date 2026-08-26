# Copyright (c) Microsoft. All rights reserved.

"""Enable ``python -m agent_framework_lab_cachebench``."""

import sys

from ._cli import main

if __name__ == "__main__":
    sys.exit(main())
