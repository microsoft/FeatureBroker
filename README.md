# Introduction 

Welcome to the Resonance project. The purpose of this project is to enable the
utilization of machine learning models across components and platforms.

It includes the notion of a feature broker, a library that enables sharing of
context/features across code components, as well as providing a common
abstraction layer for machine learning models. The core library includes the
central interfaces and abstractions. Particular model implementations are
available as specialized auxiliary libraries for:

1. Vowpal Wabbit models via [VW Slim][vwslim],
2. [ONNX models][onnxrt],
3. Azure Personalization Service (APS) models.

[vwslim]: https://github.com/VowpalWabbit/vowpal_wabbit/tree/master/vowpalwabbit/slim
[onnxrt]: https://github.com/microsoft/onnxruntime

Note that while the library was originally intended to service asynchronous
scenarios across different components, the API also supports efficient
single-component single-thread inferencing as well. The abstraction for models
is also implementable so other types of models can be plugged in to be used
through the common API beyond the examples above.

# Building

Let's imagine you've cloned the repo.

```
git submodule update --init --recursive
```

This will fetch the submodules.

You can then set up your build environment.

```
cmake . -B build.d
```

This will construct the platform specific build environment. For example, on
Windows with a properly configured Visual Studio, this will create in the
directory `build.d` the solution file `Resonance.sln`.

# Example

The following example is adapted from our `RunMatrixMultiply` test of our ONNX
model support. As the name suggests, this is a test of a model with a single
[ONNX MatMul operator][onnxMatmul] over two inputs. This class is one of two
built-in implementations of `inference::Model`. ONNX models contain within
themselves a description of their input and output schemas, including their
types. This example is structured more for simplicity of illustrating the key
structures, rather complete coverage.

[onnxMatmul]: https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul

First, we start with a pre-amble where we simply load the model.

```cpp
auto path = test_dir_path + "matmul.onnx";
auto modelExpected = onnx_model::Model::Load(path);
auto model = modelExpected.value();
```

Note that in this first phase we are loading a model. This code is peculiar to
the `onnx_model` namespace, but we already see some common conventions emerge.
In particular, throughout this codebase there is a lot of usage of things like
`tl::expected<T, std::error_code>` for code that *might* fail. (In this case,
`T` would be a `std::shared_ptr<Model>`.) One can read more about this
[structure here](https://github.com/TartanLlama/expected).

Throughout this code you'll note that we are living very dangerously, by not
actually checking to see whether the `tl::expected` actually has the values we
expect. In real production code a user would want to check and provide the
remediation that is appropriate for their library (whether that is raising an
exception, returning an error code, or something else is up to the user).

The particular model that is being loaded is an ONNX model that takes two
placeholders (in the ONNX model sense) named `A` and `B`, and produces another
output `C`, that is the result of performing a matrix multiplication of those
two inputs.

```cpp
FeatureBroker fb(model);
// Binding inputs and binding outputs.
auto inputA = fb.BindInput<Tensor<double>>("A:0").value_or(nullptr);
auto inputB = fb.BindInput<Tensor<double>>("B:0").value_or(nullptr);
auto output = fb.BindOutput<Tensor<double>>("C:0").value_or(nullptr);
```

In this particular section we create the `FeatureBroker` *and* associate it
with a model in the first step. The inputs and outputs are bound, fed, and
consumed in all the same function so it is simplest to do things that way.

In other scenarios, especially when multiple models are being used for
inference, it is often common to create a feature broker without any model at
all, bind inputs to that, and then later on the actual inferencing component
`.Fork`s another feature broker as a "child" of that feature broker, and binds
an input. But in this simple example we are simply doing everything all at
once.

Note again the usage of `tl::expected` for both the input and output pipe
bindings, since bindings can fail due to a variety of factors. (But in this
case, they do not, because the inputs and outputs happen to match the types
advertised by the model.)

```cpp
auto adata = std::shared_ptr<double>(new double[6]{1, 2, 3, 4, 5, 6});
auto bdata = std::shared_ptr<double>(new double[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
Tensor<double> a(adata, {2, 3});
Tensor<double> b(bdata, {3, 4});
inputA->Feed(a);
inputB->Feed(b);
Tensor<double> c;
auto updateExpected = output->UpdateIfChanged(c);
// This should be an tl::expected<bool, ...>, where there is no error and the value is true.
const double* cdata = c.Data();

// C should contain the values as indicated in the last value:
//
//       / 1 2 3 \ / 1  2  3  4 \   / 38 44  50  56 \
// A B = \ 4 5 6 / | 5  6  7  8 | = \ 83 98 113 128 /
//                 \ 9 10 11 12 /
```

Note that the `Feed` operation here succeeds unconditionally. This is
important. One of the chief design considerations is that components that feed
values used in inference did not want to have to deal with failure on account
of not being what the model expected, or anything else which is practically
always outside of their control. This is part of the design goal of giving
feature providing components as little trouble as possible. So long as it is
the right type, a `Feed` call will succeed. Any bad forms would have to be
detected and handled downstream by the inferencing component.

Speaking of the inferencing component, let us move on to `UpdateIfChanged`.
This method returns a `tl::expected<bool, ...>` structure. An error might
occur for any reason defined by the library. In this case, to give a specific
example, an error would arise if the bound input matrices were not of
compatible dimension.)

Note also the use of `inference::Tensor<T>` in this code. Crucially, while the
library *does* provide a simple wrapping `Tensor` object, this is not
prescriptive: the `FeatureBroker` library is unopinionated w.r.t. what types
are allowed, assuming that in the compilation of C++ used
[std::type_info](https://en.cppreference.com/w/cpp/types/type_info) is
supported.

# Contributing

This project welcomes contributions and suggestions.  Most contributions
require you to agree to a Contributor License Agreement (CLA) declaring that
you have the right to, and actually do, grant us the rights to use your
contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether
you need to provide a CLA and decorate the PR appropriately (e.g., status
check, comment). Simply follow the instructions provided by the bot. You will
only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct][msoscoc].
For more information see the [Code of Conduct FAQ][msoscoc-faq] or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional
questions or comments.

[msoscoc]: https://opensource.microsoft.com/codeofconduct/
[msoscoc-faq]: https://opensource.microsoft.com/codeofconduct/faq/