import setuptools
import nuc_chain

long_description = nuc_chain.__doc__

setuptools.setup(
    name="nuc_chain",
    version=nuc_chain.__version__,
    author="Bruno Belran",
    author_email="brunobeltran0@gmail.com",
    description="Modeling chromatin as a chain of nucleosomes",
    long_description=long_description,
    long_description_content_type="text/rst",
    url="https://gitlab.com/brunobeltran/nuc_chain",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'seaborn',
                      'sympy'],
    setup_requires=['sphinx', 'sphinx_rtd_theme'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ),
)
