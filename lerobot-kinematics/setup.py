from setuptools import setup, Extension
import os
import numpy

extra_folders = [
    "lerobot_kinematics/core",
]

def package_files(directory):
    paths = []
    for (pathhere, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", pathhere, filename))
    return paths


extra_files = []
for extra_folder in extra_folders:
    extra_files += package_files(extra_folder)

frne = Extension(
    "lerobot_kinematics.frne",
    sources=[
        "./lerobot_kinematics/core/vmath.c",
        "./lerobot_kinematics/core/ne.c",
        "./lerobot_kinematics/core/frne.c",
    ],
    include_dirs=["./lerobot_kinematics/core/"],
)

fknm = Extension(
    "lerobot_kinematics.fknm",
    sources=[
        "./lerobot_kinematics/core/methods.cpp",
        "./lerobot_kinematics/core/ik.cpp",
        "./lerobot_kinematics/core/linalg.cpp",
        "./lerobot_kinematics/core/fknm.cpp",
    ],
    include_dirs=["./lerobot_kinematics/core/", numpy.get_include()],
)

setup(
    ext_modules=[frne, fknm],
    package_data={"lerobot_kinematics": extra_files},
)