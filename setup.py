from setuptools import setup

package_name = 'proxemic_detector_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eric',
    maintainer_email='eric@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'proxemic_detector_node = proxemic_detector_pkg.proxemic_detector_node:main',
            'proxemic_detector_nodeproxemic_detector_node = proxemic_detector_pkg.proxemic_detector_nodeproxemic_detector_node:main'
        ],
    },
)
