"""
Setup of Deep LfD python codebase
Author: Michael Laskey
"""
from setuptools import setup

setup(name='il_ros_hsr',
      version='0.1.dev0',
      description='IL HSR project code',
      author='Michael Laskey',
      author_email='laskeymd@berkeley.edu',
      package_dir = {'': 'src'},
      packages=['il_ros_hsr', 'il_ros_hsr.tensor', 'il_ros_hsr.p_pi', 'il_ros_hsr.core'],
     )
