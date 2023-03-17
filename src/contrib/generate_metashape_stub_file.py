# Script generates a Metashape.pyi stub file
# which can be used for type annotations in your preferred code editor when coding Metashape scripts
#
# This is NOT a script to be used in Metashape or for processing Metashape projects
#
# For this to work, the Metashape wheel has to be installed
# Download the Python 3 module here: https://www.agisoft.com/downloads/installer/
# Then run `pip install Metashape-2.0.1-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl`
# (preferably do this in a virtual environment)
#
# Depending on your code editor
# you need to find the right location to put the resulting `Metashape.pyi` file.
# One possible location could be:
# `<your python environment>/Lib/site-packages/Metashape/Metashape.pyi``
#
# Author: arbertrary, endzeit7@gmail.com
# 2023-03-17


import importlib
import inspect
import re
import textwrap

metashape = importlib.import_module('Metashape.Metashape')
# metashape_dict = metashape.__dict__


# https://stackoverflow.com/questions/49409249/python-generate-function-stubs-from-c-module
def write_stub_recursive(name: str, object: type, level: int):
    """
    Recurse through the imported module using the inspect library
    @param name: Name of the currently inspected object
    @type str

    @param object: The currently inspected object
    @type type

    @param level: The current depth of recursion
    @type int
    """

    # Increase the indentation with each recursion
    offset = "\t"*level

    if inspect.isclass(object):
        f.write('\n')
        f.write(f'{offset}class {name}:\n')

        f.write(textwrap.indent(
            f'"""{inspect.cleandoc(object.__doc__)}"""', offset+"\t"))
        f.write('\n')

        for child_obj_name, child_obj in inspect.getmembers(object):
            if not child_obj_name.startswith('__'):
                write_stub_recursive(child_obj_name, child_obj, level + 1)
    else:
        # If it's not a class it is either a method or an attribute
        if not name.startswith('__'):
            # inspect.ismethod(object) did not seem to work so I used this slightly crude solution
            # to check if the object is a method
            if "method" in str(object):
                try:
                    f.write(
                        f'{offset}def {name} {inspect.signature(object)}:\n')
                except:
                    f.write(f'{offset}def {name} (self, *args, **kwargs):\n')
                f.write(textwrap.indent(
                    f'"""{inspect.cleandoc(object.__doc__)}"""', offset+"\t"))
                f.write(f'\n{offset}...\n')
            else:
                docstring = inspect.cleandoc(object.__doc__)

                # For the .pyi stub file to really work we need correct typing for string/str
                # Also replace the "Metashape" occurrences. Not needed for .pyi
                docstring = docstring.replace(
                    "string", "str").replace("Metashape.", "")

                # Search the docstring of the Metashape package using regex patterns
                builtin_type_match = re.search(":type:(.*)", docstring)

                # Use the "Any" type as default
                printed_type = "Any"

                if builtin_type_match:
                    builtin_type = builtin_type_match.group(1)

                    metashape_type_match = re.search(
                        ":class:`(.*?)`", builtin_type)

                    if metashape_type_match:
                        printed_type = metashape_type_match.group(1)
                    else:
                        collection_match = re.search(
                            "(\w*) of (.*)", builtin_type)
                        if collection_match:
                            printed_type = f"{collection_match.group(1)}"
                        else:
                            printed_type = builtin_type

                try:
                    f.write(
                        f'{offset}{name} {inspect.signature(object)}: {printed_type}\n')
                except:
                    f.write(f'{offset}{name}: {printed_type}\n')

                # indent the entire docstring block
                f.write(textwrap.indent(
                    f'"""{docstring}"""', offset))

                f.write(f'\n\n')


# write to file
with open('Metashape.pyi', 'w') as f:
    level = 0
    f.write("from typing import Any\n")

    # root level iteration
    for name, root_obj in inspect.getmembers(metashape):
        if inspect.isclass(root_obj):
            f.write('\n')
            f.write(f'class {name}:\n')

            for child_name, obj in inspect.getmembers(root_obj):
                if not child_name.startswith('__'):
                    write_stub_recursive(child_name, obj, level + 1)
