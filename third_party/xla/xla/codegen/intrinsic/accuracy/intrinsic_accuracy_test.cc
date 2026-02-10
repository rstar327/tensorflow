/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// HLO-level accuracy tests for unary math intrinsics.
//
// These tests operate at the HLO level, compiling and running HLO modules
// through XLA's full compilation pipeline. This means:
//   1. Tests are resilient to changes in the underlying intrinsic
//      implementations (e.g., swapping LLVM intrinsics, changing polynomial
//      approximations, etc.).
//   2. They test the actual end-to-end path that user code follows.
//   3. Accuracy regressions are caught regardless of where in the pipeline
//      the regression was introduced.
//
// Golden baselines are generated offline using mpmath at 50 digits of
// precision (see generate_golden_baselines.py).
//
// NOTE: XLA already has exhaustive tests in xla/tests/exhaustive/ that test
// every representable float value for F32 and smaller types (and sampled
// subsets for F64). Those tests compare against a reference interpreter
// backend. These golden-baseline tests complement the exhaustive suite by:
//   - Comparing against an independent high-precision reference (mpmath),
//     not XLA's own interpreter.
//   - Providing explicit ULP budgets per op that serve as a contract.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/codegen/intrinsic/accuracy/accuracy_budget.h"
#include "xla/codegen/intrinsic/accuracy/golden_baselines.h"
#include "xla/fp_util.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

constexpr int kInfiniteUlpDistance = 1000'000;

// ---------------------------------------------------------------------------
// Test fixture: uses PjRt test runner.
// ---------------------------------------------------------------------------

using HloIntrinsicAccuracyTest = HloPjRtTestBase;

// ---------------------------------------------------------------------------
// Accuracy reporting (ULP-based, independent of XLA's ErrorSpec).
// ---------------------------------------------------------------------------

struct AccuracyReport {
  int max_ulp_error = 0;
  double mean_ulp_error = 0.0;
  int count = 0;
  double worst_input = 0.0;
  double worst_expected = 0.0;
  double worst_actual = 0.0;
};

template <typename T>
int UlpDistance(T actual, T expected) {
  if (std::isnan(expected)) {
    return std::isnan(actual) ? 0 : kInfiniteUlpDistance;
  }
  if (std::isinf(expected)) {
    return (std::isinf(actual) &&
            std::signbit(expected) == std::signbit(actual))
               ? 0
               : kInfiniteUlpDistance;
  }
  if (std::isnan(actual) || std::isinf(actual)) {
    return kInfiniteUlpDistance;
  }
  return std::abs(CalculateDistanceInFloats(actual, expected));
}

template <typename T>
AccuracyReport ComputeAccuracyReport(
    absl::Span<const codegen::intrinsic::accuracy::RefPoint> golden,
    const T* results, size_t count) {
  AccuracyReport report;
  int64_t total_ulp = 0;

  for (size_t i = 0; i < count; ++i) {
    T expected = static_cast<T>(golden[i].expected);

    // Skip subnormals in expected values — they are platform-dependent.
    if (std::fpclassify(expected) == FP_SUBNORMAL) continue;

    T actual = results[i];
    int ulp = UlpDistance(actual, expected);
    total_ulp += ulp;
    report.count++;

    if (ulp > report.max_ulp_error) {
      report.max_ulp_error = ulp;
      report.worst_input = golden[i].input;
      report.worst_expected = golden[i].expected;
      report.worst_actual = static_cast<double>(actual);
    }
  }

  if (report.count > 0) {
    report.mean_ulp_error = static_cast<double>(total_ulp) / report.count;
  }
  return report;
}

void LogAccuracyReport(const AccuracyReport& report,
                       absl::string_view test_name) {
  LOG(INFO) << "Accuracy Report for " << test_name << ":\n"
            << "  Tested points: " << report.count << "\n"
            << "  Max ULP Error: " << report.max_ulp_error << "\n"
            << "  Mean ULP Error: " << report.mean_ulp_error << "\n"
            << "  Worst Case: input=" << report.worst_input
            << ", expected=" << report.worst_expected
            << ", actual=" << report.worst_actual;
}

// ---------------------------------------------------------------------------
// HLO module templates.
// ---------------------------------------------------------------------------

std::string MakeUnaryHloModule(absl::string_view op_name, PrimitiveType type,
                               int64_t count) {
  std::string type_str = (type == F32) ? "f32" : "f64";
  return absl::StrFormat(R"(
HloModule %s_accuracy_test

ENTRY main {
  input = %s[%d] parameter(0)
  ROOT result = %s[%d] %s(input)
}
)",
                         op_name, type_str, count, type_str, count, op_name);
}

// ---------------------------------------------------------------------------
// Parameterized test infrastructure.
// ---------------------------------------------------------------------------

struct IntrinsicAccuracyTestParam {
  std::string name;
  std::string hlo_op_name;
  PrimitiveType primitive_type;
  const codegen::intrinsic::accuracy::RefPoint* golden_data;
  size_t golden_count;
  int ulp_budget;
  bool fast_math = false;
};

class Param {
 public:
  explicit Param(absl::string_view op_name) : op_name_(op_name) {}

  IntrinsicAccuracyTestParam F32(
      absl::Span<const codegen::intrinsic::accuracy::RefPoint> golden,
      int ulp_budget) const {
    return Build("F32", PrimitiveType::F32, golden, ulp_budget);
  }

  IntrinsicAccuracyTestParam F64(
      absl::Span<const codegen::intrinsic::accuracy::RefPoint> golden,
      int ulp_budget) const {
    return Build("F64", PrimitiveType::F64, golden, ulp_budget);
  }

 private:
  IntrinsicAccuracyTestParam Build(
      absl::string_view suffix, PrimitiveType type,
      absl::Span<const codegen::intrinsic::accuracy::RefPoint> golden,
      int ulp_budget) const {
    // Capitalize first letter for test name aesthetics (e.g. "tanh" -> "Tanh")
    std::string name = std::string(op_name_);
    if (!name.empty()) name[0] = std::toupper(name[0]);
    // GTest parameter names must be alphanumeric (or underscore).
    std::replace_if(
        name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');

    return IntrinsicAccuracyTestParam{.name = absl::StrCat(name, suffix),
                                      .hlo_op_name = std::string(op_name_),
                                      .primitive_type = type,
                                      .golden_data = golden.data(),
                                      .golden_count = golden.size(),
                                      .ulp_budget = ulp_budget};
  }

  absl::string_view op_name_;
};

// Filter golden points: remove those whose input overflows when cast to T.
template <typename T>
std::vector<codegen::intrinsic::accuracy::RefPoint> FilterGoldenForType(
    absl::Span<const codegen::intrinsic::accuracy::RefPoint> data,
    size_t count) {
  std::vector<codegen::intrinsic::accuracy::RefPoint> filtered;
  filtered.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    T input = static_cast<T>(data[i].input);
    // Skip points where the input overflows to inf during cast.
    if (std::isinf(input) && !std::isinf(data[i].input)) continue;
    filtered.push_back(data[i]);
  }
  return filtered;
}

class HloIntrinsicAccuracyParamTest
    : public HloPjRtTestBase,
      public ::testing::WithParamInterface<IntrinsicAccuracyTestParam> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloRunnerAgnosticTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_enable_fast_math(GetParam().fast_math);
    debug_options.set_xla_cpu_enable_fast_min_max(GetParam().fast_math);
    debug_options.set_xla_gpu_enable_fast_min_max(GetParam().fast_math);
    return debug_options;
  }

 protected:
  template <typename T>
  void RunAccuracyTest(const IntrinsicAccuracyTestParam& param) {
    auto full_golden = FilterGoldenForType<T>(
        absl::MakeSpan(param.golden_data, param.golden_count),
        param.golden_count);

    std::vector<T> inputs;
    std::vector<codegen::intrinsic::accuracy::RefPoint> golden;
    inputs.reserve(full_golden.size());
    golden.reserve(full_golden.size());

    for (const auto& point : full_golden) {
      T input = static_cast<T>(point.input);
      if (param.hlo_op_name == "sqrt") {
        // Skip negative subnormals (flush to -0 != NaN)
        if (input < 0 && std::fpclassify(input) == FP_SUBNORMAL) continue;
        // Skip -0.0 if expected is NaN (from negative double subnormal)
        if (input == 0.0 && std::signbit(input) && std::isnan(point.expected)) {
          continue;
        }
      } else if (param.hlo_op_name == "rsqrt") {
        if (std::fpclassify(input) == FP_SUBNORMAL) continue;
      }
      inputs.push_back(input);
      golden.push_back(point);
    }
    auto input_literal = LiteralUtil::CreateR1<T>(inputs);
    int64_t count = inputs.size();

    std::string hlo =
        MakeUnaryHloModule(param.hlo_op_name, param.primitive_type, count);
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    module->mutable_config().set_debug_options(GetDebugOptionsForTest());

    TF_ASSERT_OK_AND_ASSIGN(auto result,
                            Execute(std::move(module), {&input_literal}));

    auto result_data = result.template data<T>();
    auto report = ComputeAccuracyReport<T>(golden, result_data.data(),
                                           result_data.size());
    LogAccuracyReport(report, param.name);

    EXPECT_LE(report.max_ulp_error, param.ulp_budget)
        << "Max ULP error " << report.max_ulp_error << " exceeds budget "
        << param.ulp_budget << ". Worst case: input=" << report.worst_input
        << ", expected=" << report.worst_expected
        << ", actual=" << report.worst_actual;
  }
};

TEST_P(HloIntrinsicAccuracyParamTest, WithinUlpBudget) {
  const auto& param = GetParam();

  if (param.primitive_type == F32) {
    RunAccuracyTest<float>(param);
  } else if (param.primitive_type == F64) {
    RunAccuracyTest<double>(param);
  } else {
    GTEST_SKIP() << "Unsupported type";
  }
}

// ---------------------------------------------------------------------------
// Test case registration.
// ---------------------------------------------------------------------------

namespace accuracy = ::xla::codegen::intrinsic::accuracy;

INSTANTIATE_TEST_SUITE_P(
    UnaryIntrinsics, HloIntrinsicAccuracyParamTest,
    ::testing::Values(
        // Tanh
        Param("tanh").F32(accuracy::kGoldenTanh, accuracy::kTanhF32MaxUlp),
        Param("tanh").F64(accuracy::kGoldenTanh, accuracy::kTanhF64MaxUlp),

        // Exp
        Param("exponential").F32(accuracy::kGoldenExp, accuracy::kExpF32MaxUlp),
        Param("exponential").F64(accuracy::kGoldenExp, accuracy::kExpF64MaxUlp),

        // Log1p
        Param("log-plus-one")
            .F32(accuracy::kGoldenLog1p, accuracy::kLog1pF32MaxUlp),
        Param("log-plus-one")
            .F64(accuracy::kGoldenLog1p, accuracy::kLog1pF64MaxUlp),

        // Rsqrt
        Param("rsqrt").F32(accuracy::kGoldenRsqrt, accuracy::kRsqrtF32MaxUlp),
        Param("rsqrt").F64(accuracy::kGoldenRsqrt, accuracy::kRsqrtF64MaxUlp),

        // Sqrt
        Param("sqrt").F32(accuracy::kGoldenSqrt, accuracy::kSqrtF32MaxUlp),
        Param("sqrt").F64(accuracy::kGoldenSqrt, accuracy::kSqrtF64MaxUlp),

        // Erf
        Param("erf").F32(accuracy::kGoldenErf, accuracy::kErfF32MaxUlp),
        Param("erf").F64(accuracy::kGoldenErf, accuracy::kErfF64MaxUlp)),
    [](const ::testing::TestParamInfo<IntrinsicAccuracyTestParam>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace xla
