from setuptools import Command, find_packages, setup
from pathlib import Path

# In[ ]:


__lib_name__ = "spiral"
__lib_version__ = "1.0"
__description__ = "Single cell integration by disentangle represent learning."
__url__ = "https://github.com/guott15/SCIDRL"
__author__ = "Tiantian Guo"
__author_email__ = "guott18@mails.tsinghua.edu.cn"
__license__ = "MIT"
__keywords__ = ["scRNA-seq", "Deep learning", "Batch effect", "Domain Adaptation"]
# __requires__ = [l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()]
__requires__ = ["requests",]


with open("README.md", "r") as fh:
    __long_description__ = fh.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['scidrl'],
    install_requires = __requires__,
    zip_safe = False,
    long_description = __long_description__,
    python_requires='>=3.6'
)
