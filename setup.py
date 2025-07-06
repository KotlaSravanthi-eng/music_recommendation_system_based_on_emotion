import os
from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    ''''
    This function returns the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() != '-e .']
    return requirements 

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'music_recommendation_system_based_on_emotion',
    version = '0.1.0',
    author= 'Kotla Sravanthi',
    author_email= 'kotlasravanthi229@gmail.com',
    packages= find_packages(),
    description= 'A content-based music recommendation system that suggests songs based on user mood using NLP and cosine similarity.',
    long_description= open('README.md').read(),
    install_requires = get_requirements('requirements.txt'),
    url= 'https://github.com/KotlaSravanthi-eng/music_recommendation_system_based_on_emotion'
)
    