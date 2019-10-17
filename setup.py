from setuptools import setup, find_packages

setup_args = dict(
    name = 'pmgtopdb',
	description = 'assorted routine utilities',
    include_package_data = True,
    packages = find_packages()
)

if __name__ == '__main__':
    print(find_packages())
    setup(**setup_args)

