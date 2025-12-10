from setuptools import find_packages, setup

package_name = 'env_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liumuyuan',
    maintainer_email='liumuyuan@jushenzhiren.com',
    description='TODO: Package description',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'env_bridge = env_bridge.env_bridge:main'
        ],
    },
)
