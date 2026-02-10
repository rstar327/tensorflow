# Copyright 2025 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("//xla:pytype.bzl", "pytype_strict_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "golden_baselines",
    hdrs = ["golden_baselines.h"],
)

cc_library(
    name = "accuracy_budget",
    hdrs = ["accuracy_budget.h"],
)

xla_cc_test(
    name = "intrinsic_accuracy_test",
    srcs = ["intrinsic_accuracy_test.cc"],
    deps = [
        ":accuracy_budget",
        ":golden_baselines",
        "//xla:fp_util",
        "//xla:literal_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/tests:client_library_test_base",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:pjrt_cpu_client_registry",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

pytype_strict_library(
    name = "generate_golden_baselines",
    srcs = ["generate_golden_baselines.py"],
    deps = ["//third_party/py/mpmath"],
)
