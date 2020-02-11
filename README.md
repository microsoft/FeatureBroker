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