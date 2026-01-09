'''
SimBEV: A Synthetic Multi-Task Multi-Sensor Driving Data Generation Tool

Copyright © 2026 Goodarz Mehr
'''

from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Read long description from README.
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Read requirements.
with open('requirements.txt', 'r', encoding='utf-8') as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# CUDA extensions
bbox_cuda_ext = CUDAExtension(
    name='simbev_tools.bbox_cuda',
    sources=[
        'simbev_tools/cuda_extensions/bbox_cuda.cpp',
        'simbev_tools/cuda_extensions/bbox_cuda_kernel.cu',
    ],
    extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': ['-O3', '--use_fast_math']
    }
)

fill_voxel_cuda_ext = CUDAExtension(
    name='simbev_tools.fill_voxel_cuda',
    sources=[
        'simbev_tools/cuda_extensions/fill_voxel_cuda.cpp',
        'simbev_tools/cuda_extensions/fill_voxel_cuda_kernel.cu',
    ],
    extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': ['-O3', '--use_fast_math']
    }
)

setup(
    name='simbev',
    version='3.0.0',
    author='Goodarz Mehr',
    author_email='goodarzm@vt.edu',
    description='Synthetic Multi-Task Multi-Sensor Driving Data Generation Tool',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GoodarzMehr/SimBEV',
    project_urls={
        'Bug Tracker': 'https://github.com/GoodarzMehr/SimBEV/issues',
        'Documentation': 'https://simbev.org',
        'Paper': 'https://arxiv.org/abs/2502.01894',
        'Dataset': 'https://drive.google.com/drive/folders/14MytQeGmW80Btg_AGPNrE18ZLdLzyGx5'
    },
    packages=find_packages(exclude=['configs', 'assets']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    ext_modules=[bbox_cuda_ext, fill_voxel_cuda_ext],
    cmdclass={
        'build_ext': BuildExtension
    },
    entry_points={
        'console_scripts': [
            'simbev=simbev.simbev:entry',
            'simbev-postprocess=simbev_tools.post_processing:entry',
            'simbev-visualize=simbev_tools.visualization:entry',
        ],
    },
    include_package_data=True,
    package_data={
        'simbev_tools': ['cuda_extensions/*.cu', 'cuda_extensions/*.cpp'],
    },
)
