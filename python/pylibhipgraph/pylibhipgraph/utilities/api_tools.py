# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import functools
import inspect
import types
import warnings

experimental_prefix = "EXPERIMENTAL"


def experimental_warning_wrapper(obj, obj_namespace_name=None):
    """
    Wrap obj in a function or class that prints a warning about it being
    "experimental" (ie. it is in the public API but subject to change or
    removal), prior to calling obj and returning its value.

    The object's name used in the warning message also has any leading __
    and/or EXPERIMENTAL string are removed from the name used in warning
    messages. This allows an object to be named with a "private" name in the
    public API so it can remain hidden while it is still experimental, but
    have a public name within the experimental namespace so it can be easily
    discovered and used.

    obj_namespace_name can be passed in to make the warning message
    clearer. For example, if the obj is the function sssp and is accessed as
    part of a package like "pylibhipgraph.experimental.sssp", obj_namespace_name
    should be "pylibhipgraph.experimental".  If obj_namespace_name is not
    passed, the namespace will be found using inspect.stack(), which can be
    expensive.
    """
    obj_type = type(obj)
    if not callable(obj):
        raise TypeError("obj must be a class or a function type, got " f"{obj_type}")

    obj_name = obj.__name__
    obj_name = obj_name.lstrip(experimental_prefix)
    obj_name = obj_name.lstrip("__")

    if obj_namespace_name is None:
        # Assume the caller of this function is the module containing the
        # experimental obj and try to get its namespace name. Default to no
        # namespace name if it could not be found.
        call_stack = inspect.stack()
        calling_frame = call_stack[1].frame
        obj_namespace_name = calling_frame.f_locals.get("__name__")

    dot = "." if obj_namespace_name is not None else ""
    warning_msg = (
        f"{obj_namespace_name}{dot}{obj_name} is experimental and may "
        "change or be removed in a future release."
    )

    # If obj is a class, create a wrapper class which 1) inherits from the
    # incoming class, and 2) has a ctor that simply prints the warning and
    # calls the base class ctor. A wrapper class is needed so the new type
    # matches the incoming type.
    # Ideally a wrapper function would be created and assigned to the class as
    # the new __init__, but #2 is necessary since assigning attributes cannot
    # be done to a builtin type (such as a class defined in cython).
    if obj_type is type:

        class WarningWrapperClass(obj):
            def __init__(self, *args, **kwargs):
                warnings.warn(warning_msg, PendingDeprecationWarning)
                # call base class __init__ for python, but cython classes do
                # not have a standard callable __init__ and assigning to self
                # works instead.
                if isinstance(obj.__init__, types.FunctionType):
                    super(WarningWrapperClass, self).__init__(*args, **kwargs)
                else:
                    self = obj(*args, **kwargs)

        WarningWrapperClass.__module__ = obj_namespace_name
        WarningWrapperClass.__qualname__ = obj_name
        WarningWrapperClass.__name__ = obj_name
        WarningWrapperClass.__doc__ = obj.__doc__

        return WarningWrapperClass

    # If this point is reached, the incoming obj is a function so simply wrap
    # it and return the wrapper. Since the wrapper is a function type, it will
    # match the incoming obj type.
    @functools.wraps(obj)
    def warning_wrapper_function(*args, **kwargs):
        warnings.warn(warning_msg, PendingDeprecationWarning)
        return obj(*args, **kwargs)

    warning_wrapper_function.__module__ = obj_namespace_name
    warning_wrapper_function.__qualname__ = obj_name
    warning_wrapper_function.__name__ = obj_name
    warning_wrapper_function.__doc__ = obj.__doc__

    return warning_wrapper_function


def promoted_experimental_warning_wrapper(obj, obj_namespace_name=None):
    """
    Wrap obj in a function of class that prints a warning about it being
    close to being removed, prior to calling obj and returning its value.

    This is different from experimental_warning_wrapper in that the object
    has been promoted out of EXPERIMENTAL and thus has two versions of the
    same object. This wrapper is applied to the one with the "private" name,
    urging the user to instead use the one in the public API, which does not
    have the experimental namespace.

    obj_namespace_name can be passed in to make the warning message
    clearer. For example, if the obj is the function sssp and is accessed as
    part of a package like "pylibhipgraph.experimental.sssp", obj_namespace_name
    should be "pylibhipgraph.experimental".  If obj_namespace_name is not
    passed, the namespace will be found using inspect.stack(), which can be
    expensive.
    """
    obj_type = type(obj)
    if not callable(obj):
        raise TypeError("obj must be a class or a function type, got " f"{obj_type}")

    obj_name = obj.__name__
    obj_name = obj_name.lstrip(experimental_prefix)
    obj_name = obj_name.lstrip("__")

    if obj_namespace_name is None:
        # Assume the caller of this function is the module containing the
        # experimental obj and try to get its namespace name. Default to no
        # namespace name if it could not be found.
        call_stack = inspect.stack()
        calling_frame = call_stack[1].frame
        obj_namespace_name = calling_frame.f_locals.get("__name__")

    dot = "." if obj_namespace_name is not None else ""
    warning_msg = (
        f"{obj_namespace_name}{dot}{obj_name} has been promoted out of "
        "experimental. Use the non-experimental version instead, "
        "as this one will be removed in a future release."
    )

    if obj_type is type:

        class WarningWrapperClass(obj):
            def __init__(self, *args, **kwargs):
                warnings.warn(warning_msg, DeprecationWarning)
                # call base class __init__ for python, but cython classes do
                # not have a standard callable __init__ and assigning to self
                # works instead.
                if isinstance(obj.__init__, types.FunctionType):
                    super(WarningWrapperClass, self).__init__(*args, **kwargs)
                else:
                    self = obj(*args, **kwargs)

        WarningWrapperClass.__module__ = obj_namespace_name
        WarningWrapperClass.__qualname__ = obj_name
        WarningWrapperClass.__name__ = obj_name

        return WarningWrapperClass

    @functools.wraps(obj)
    def warning_wrapper_function(*args, **kwargs):
        warnings.warn(warning_msg, DeprecationWarning)
        return obj(*args, **kwargs)

    warning_wrapper_function.__module__ = obj_namespace_name
    warning_wrapper_function.__qualname__ = obj_name
    warning_wrapper_function.__name__ = obj_name

    return warning_wrapper_function


def deprecated_warning_wrapper(obj, obj_namespace_name=None):
    """
    Wrap obj in a function or class that prints a warning about it being
    deprecated (ie. it is in the public API but will be removed or replaced
    by a refactored version), prior to calling obj and returning its value.

    obj_namespace_name can be passed in to make the warning message
    clearer. For example, if the obj is the function sssp and is accessed as
    part of a package like "pylibhipgraph.experimental.sssp", obj_namespace_name
    should be "pylibhipgraph.experimental".  If obj_namespace_name is not
    passed, the namespace will be found using inspect.stack(), which can be
    expensive.
    """
    obj_type = type(obj)
    if not callable(obj):
        raise TypeError("obj must be a class or a function type, got " f"{obj_type}")

    obj_name = obj.__name__
    if obj_namespace_name is None:
        # Assume the caller of this function is the module containing the
        # deprecated obj and try to get its namespace name. Default to no
        # namespace name if it could not be found.
        call_stack = inspect.stack()
        calling_frame = call_stack[1].frame
        obj_namespace_name = calling_frame.f_locals.get("__name__")

    dot = "." if obj_namespace_name is not None else ""
    warning_msg = (
        f"{obj_namespace_name}{dot}{obj_name} has been deprecated and will "
        "be removed in a future release."
    )

    if obj_type is type:

        class WarningWrapperClass(obj):
            def __init__(self, *args, **kwargs):
                warnings.warn(warning_msg, DeprecationWarning)
                # call base class __init__ for python, but cython classes do
                # not have a standard callable __init__ and assigning to self
                # works instead.
                if isinstance(obj.__init__, types.FunctionType):
                    super(WarningWrapperClass, self).__init__(*args, **kwargs)
                else:
                    self = obj(*args, **kwargs)

        WarningWrapperClass.__module__ = obj_namespace_name
        WarningWrapperClass.__qualname__ = obj_name
        WarningWrapperClass.__name__ = obj_name

        return WarningWrapperClass

    @functools.wraps(obj)
    def warning_wrapper_function(*args, **kwargs):
        warnings.warn(warning_msg, DeprecationWarning)
        return obj(*args, **kwargs)

    warning_wrapper_function.__module__ = obj_namespace_name
    warning_wrapper_function.__qualname__ = obj_name
    warning_wrapper_function.__name__ = obj_name

    return warning_wrapper_function
