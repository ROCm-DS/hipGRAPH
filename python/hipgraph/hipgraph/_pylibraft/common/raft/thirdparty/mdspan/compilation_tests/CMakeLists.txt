# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

macro(add_compilation_test name)
    if(MDSPAN_TEST_LANGUAGE)
        set_source_files_properties(${name} PROPERTIES LANGUAGE ${MDSPAN_TEST_LANGUAGE})
    endif()
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} mdspan)
endmacro()

add_compilation_test(ctest_constructor_sfinae)
add_compilation_test(ctest_extents_ctors)
add_compilation_test(ctest_extents_type_check)
add_compilation_test(ctest_layout_convertible)
add_compilation_test(ctest_mdspan_convertible)
add_compilation_test(ctest_standard_layout)
add_compilation_test(ctest_trivially_copyable)
if(NOT MDSPAN_ENABLE_CUDA)
    add_compilation_test(ctest_no_unique_address)
    add_compilation_test(ctest_compressed_pair_layout)
endif()
add_compilation_test(ctest_constexpr_dereference)
add_compilation_test(ctest_constexpr_submdspan)
add_compilation_test(ctest_constexpr_layouts)
